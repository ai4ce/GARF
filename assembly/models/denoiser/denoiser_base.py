"""
Adapted from puzzlefusion++
https://github.com/eric-zqwang/puzzlefusion-plusplus
"""

import os
import json
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import pytorch3d.transforms as transforms
import trimesh
from scipy.spatial.transform import Rotation as R
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from diffusers import SchedulerMixin
from .modules.evaluation.evaluator import (
    calc_part_acc,
    calc_part_acc_weighted,
    calc_shape_cd,
    calc_shape_cd_weighted,
    rot_metrics,
    trans_metrics,
)


class DenoiserBase(L.LightningModule):
    def __init__(
        self,
        feature_extractor_ckpt: str,
        feature_extractor: L.LightningModule,
        denoiser: nn.Module,
        noise_scheduler: SchedulerMixin,
        val_noise_scheduler: SchedulerMixin,
        optimizer: "partial[torch.optim.Optimizer]",
        lr_scheduler: "partial[torch.optim.lr_scheduler._LRScheduler]" = None,
        inference_config: dict = None,
        lora_config: LoraConfig = None,
        **kwargs,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.denoiser = denoiser
        self.noise_scheduler = noise_scheduler
        self.val_noise_scheduler = val_noise_scheduler
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.inference_config = inference_config or dict()
        self.lora_config = lora_config

        if feature_extractor_ckpt is not None:
            self.feature_extractor.load_state_dict(
                torch.load(
                    feature_extractor_ckpt,
                    map_location="cpu",
                    weights_only=True,
                )["state_dict"]
            )

        # Set feature_extractor to eval mode and freeze parameters
        self.feature_extractor = self.feature_extractor.eval()
        for module in self.feature_extractor.modules():
            module.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.rmse_r_list = []
        self.rmse_t_list = []
        self.acc_list = []
        self.cd_list = []

    def enable_lora(
        self,
        ckpt_path: Optional[str] = None,
    ):
        if ckpt_path is not None:
            checkpoint = torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
            self.lora_config = checkpoint["lora_config"]
            self.denoiser = get_peft_model(self.denoiser, self.lora_config)
            set_peft_model_state_dict(self.denoiser, checkpoint["state_dict"])
            return

        if self.lora_config is not None:
            self.denoiser = get_peft_model(self.denoiser, self.lora_config)

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is not None or isinstance(self.denoiser, PeftModel):
            checkpoint["lora_config"] = self.lora_config
            checkpoint["state_dict"] = get_peft_model_state_dict(self.denoiser)

        return super().on_save_checkpoint(checkpoint)

    def _extract_features(self, data_dict: dict):
        points = self.feature_extractor(data_dict)["point"]
        points["batch"] = points["batch"].clone()
        return points

    @staticmethod
    def se3_to_matrix(transform: torch.Tensor) -> torch.Tensor:
        """Convert SE(3) 7D vector [t, quat] to a 4x4 matrix."""
        if transform.ndim == 1:
            transform = transform.unsqueeze(0)

        trans = transform[..., :3].to(dtype=torch.float32)
        quat = transform[..., 3:].to(dtype=torch.float32)
        rot_mat = transforms.quaternion_to_matrix(quat)

        batch_shape = rot_mat.shape[:-2]
        eye = torch.eye(4, device=transform.device, dtype=rot_mat.dtype)
        eye = eye.expand(*batch_shape, 4, 4).clone()
        eye[..., :3, :3] = rot_mat
        eye[..., :3, 3] = trans
        return eye.squeeze(0)

    def log_metrics(
        self,
        metrics: dict,
        prefix: str = "",
    ):
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{prefix}/{metric_name}",
                metric_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

    def training_step(self, data_dict: dict):
        output_dict = self(data_dict)
        loss_dict, counted_losses = self._loss(data_dict, output_dict)

        total_loss = 0
        for loss_name, loss_value in loss_dict.items():
            if loss_name in counted_losses:
                total_loss += loss_value

        self.log_metrics(loss_dict, prefix="train")
        return total_loss

    def validation_step(self, data_dict: dict):
        output_dict = self(data_dict)
        loss_dict, _ = self._loss(data_dict, output_dict)
        self.log_metrics(loss_dict, prefix="val")

        # Full Evaluation
        points_per_part = data_dict["points_per_part"]
        part_valids = points_per_part != 0
        part_scale = data_dict["scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        pts = data_dict["pointclouds"]
        B, P = points_per_part.shape

        gt_trans = data_dict["translations"][part_valids]  # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]  # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        noisy_trans_and_rots = torch.randn(
            gt_trans_and_rots.shape, device=self.device
        )  # (valid_P, 7)
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0)).as_quat()).float().to(self.device)
        )[..., [3, 0, 1, 2]]
        noisy_trans_and_rots[..., 3:] = noise_rots

        # Overriding the reference part with the ground truth
        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]
        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        all_steps_preds = []

        # Extracting features
        latent = self._extract_features(data_dict)
        self.val_noise_scheduler.set_timesteps(
            num_inference_steps=self.inference_config.get("num_inference_steps", 20)
        )

        # Denoising
        for t in self.val_noise_scheduler.timesteps:
            timesteps = t.reshape(-1).repeat(len(noisy_trans_and_rots)).to(self.device)
            denoiser_out = self.denoiser(
                x=noisy_trans_and_rots,
                timesteps=timesteps,
                latent=latent,
                part_valids=part_valids,
                scale=part_scale,
                ref_part=ref_part,
            )
            model_pred = denoiser_out["pred"]
            noisy_trans_and_rots = self.val_noise_scheduler.step(
                model_pred, t, noisy_trans_and_rots
            ).prev_sample
            noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                dtype=noisy_trans_and_rots.dtype
            )
            all_steps_preds.append(noisy_trans_and_rots.clone())

        pred_trans = noisy_trans_and_rots[..., :3].detach()
        pred_rots = noisy_trans_and_rots[..., 3:].detach()

        # Recover SE3 back to padded mode
        pred_trans_padded = torch.zeros(
            (B, P, 3), device=pred_trans.device, dtype=pred_trans.dtype
        )
        pred_rots_padded = torch.zeros(
            (B, P, 4), device=pred_rots.device, dtype=pred_rots.dtype
        )
        gt_trans_padded = torch.zeros(
            (B, P, 3), device=gt_trans.device, dtype=pred_trans.dtype
        )
        gt_rots_padded = torch.zeros(
            (B, P, 4), device=gt_rots.device, dtype=pred_rots.dtype
        )
        pred_trans_padded[part_valids] = pred_trans
        pred_rots_padded[part_valids] = pred_rots
        gt_trans_padded[part_valids] = gt_trans.to(dtype=gt_trans_padded.dtype)
        gt_rots_padded[part_valids] = gt_rots.to(dtype=gt_rots_padded.dtype)

        # Two scenarios: (B, P, N, 3) or (B, N_sum, 3)
        # First one is for uniform sampling, second one is for weighted sampling
        # We have to calculate shape_cd and part_acc differently

        # Uniform sampling
        if pts.ndim == 4:
            B, P, N, C = pts.shape
            # (B, P, N, 1)
            expanded_part_scale = data_dict["scale"].unsqueeze(-1).expand(-1, -1, N, -1)
            pts = pts * expanded_part_scale  # (B, P, N, 3)

            acc, _, _ = calc_part_acc(
                pts,
                trans1=pred_trans_padded,
                trans2=gt_trans_padded,
                rot1=pred_rots_padded,
                rot2=gt_rots_padded,
                valids=part_valids,
                points_per_part=points_per_part,
            )

            shape_cd = calc_shape_cd(
                pts,
                trans1=pred_trans_padded,
                trans2=gt_trans_padded,
                rot1=pred_rots_padded,
                rot2=gt_rots_padded,
                valids=part_valids,
                points_per_part=points_per_part,
            )
        else:
            B, N_sum, C = pts.shape
            scale = data_dict["scale"][part_valids]
            scale = scale.repeat_interleave(points_per_part[part_valids], dim=0)
            pts = (pts.view(-1, C) * scale).view(B, N_sum, C)

            # Calculate Part Acc
            acc = calc_part_acc_weighted(
                pts,
                gt_trans=gt_trans,
                gt_rots=gt_rots,
                pred_trans=pred_trans,
                pred_rots=pred_rots,
                points_per_part=points_per_part,
                part_valids=part_valids,
                part_valids_wo_redundancy=part_valids,
            )

            # Calculate Shape Chamfer Distance
            shape_cd = calc_shape_cd_weighted(
                pts,
                gt_trans=gt_trans,
                gt_rots=gt_rots,
                pred_trans=pred_trans,
                pred_rots=pred_rots,
                points_per_part=points_per_part,
                part_valids=part_valids,
                part_valids_wo_redundancy=part_valids,
            )

        rmse_r = rot_metrics(pred_rots_padded, gt_rots_padded, part_valids, "rmse")
        rmse_t = trans_metrics(pred_trans_padded, gt_trans_padded, part_valids, "rmse")

        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)

        return output_dict

    def test_step(self, data_dict, idx):
        output_dict = self(data_dict)
        loss_dict, _ = self._loss(data_dict, output_dict)
        self.log_metrics(loss_dict, prefix="test")

        points_per_part = data_dict["points_per_part"]
        part_valids = points_per_part != 0
        part_scale = data_dict["scale"][part_valids]  # (valid_P, 1)
        ref_part = data_dict["ref_part"][part_valids]  # (valid_P,)
        if self.inference_config.get("anchor_free", False):
            ref_part = torch.zeros_like(ref_part, dtype=torch.bool)
        pts = data_dict["pointclouds"]
        B, P = points_per_part.shape

        gt_trans = data_dict["translations"][part_valids]  # (valid_P, 3)
        gt_rots = data_dict["quaternions"][part_valids]  # (valid_P, 4)
        gt_trans_and_rots = torch.cat([gt_trans, gt_rots], dim=-1)  # (valid_P, 7)

        noisy_trans_and_rots = torch.randn(
            gt_trans_and_rots.shape, device=self.device
        )  # (valid_P, 7)
        noise_rots = (
            torch.tensor(R.random(gt_rots.size(0)).as_quat()).float().to(self.device)
        )[..., [3, 0, 1, 2]]
        noisy_trans_and_rots[..., 3:] = noise_rots

        reference_gt_and_rots = torch.zeros_like(gt_trans_and_rots, device=self.device)
        reference_gt_and_rots[ref_part] = gt_trans_and_rots[ref_part]

        num_parts_cum = nn.functional.pad(
            torch.cumsum(data_dict["num_parts"], dim=-1), (1, 0), value=0
        )
        all_steps_preds = []

        noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part]

        noisy_trans_and_rots = noisy_trans_and_rots.half()
        reference_gt_and_rots = reference_gt_and_rots.half()
        gt_trans_and_rots = gt_trans_and_rots.half()

        for iter in range(self.inference_config.get("max_iters", 1)):
            latent = self._extract_features(data_dict)

            if self.inference_config.get("one_step_init", False) and iter == 0:
                self.val_noise_scheduler.set_timesteps(1)
                t = self.val_noise_scheduler.timesteps[0]
                timesteps = (
                    t.reshape(-1)
                    .repeat(len(noisy_trans_and_rots))
                    .to(noisy_trans_and_rots.device)
                )
                denoiser_out = self.denoiser(
                    x=noisy_trans_and_rots,
                    timesteps=timesteps,
                    latent=latent,
                    part_valids=part_valids,
                    scale=part_scale,
                    ref_part=ref_part,
                )
                model_pred = denoiser_out["pred"]

                noisy_trans_and_rots = self.val_noise_scheduler.step(
                    model_pred, t, noisy_trans_and_rots
                ).prev_sample  # (valid_P, 7)
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                    dtype=noisy_trans_and_rots.dtype
                )
                all_steps_preds.append(noisy_trans_and_rots.clone())

            self.val_noise_scheduler.set_timesteps(
                num_inference_steps=self.inference_config.get("num_inference_steps", 20)
            )
            for t in self.val_noise_scheduler.timesteps:
                timesteps = (
                    t.reshape(-1).repeat(len(noisy_trans_and_rots)).to(self.device)
                )
                denoiser_out = self.denoiser(
                    x=noisy_trans_and_rots,
                    timesteps=timesteps,
                    latent=latent,
                    part_valids=part_valids,
                    scale=part_scale,
                    ref_part=ref_part,
                )
                model_pred = denoiser_out["pred"]

                noisy_trans_and_rots = self.val_noise_scheduler.step(
                    model_pred, t, noisy_trans_and_rots
                ).prev_sample  # (valid_P, 7)
                noisy_trans_and_rots[ref_part] = reference_gt_and_rots[ref_part].to(
                    dtype=noisy_trans_and_rots.dtype
                )
                all_steps_preds.append(noisy_trans_and_rots.clone())

        pred_trans = noisy_trans_and_rots[..., :3].detach()  # (valid_P, 3)
        pred_rots = noisy_trans_and_rots[..., 3:].detach()  # (valid_P, 4)

        if self.inference_config.get("anchor_free", False):
            ref_part = data_dict["ref_part"][part_valids]
            ref_part_gt_trans = gt_trans_and_rots[ref_part, :3]
            ref_part_gt_quat = gt_trans_and_rots[ref_part, 3:]
            ref_part_pred_trans = noisy_trans_and_rots[ref_part, :3]
            ref_part_pred_quat = noisy_trans_and_rots[ref_part, 3:]
            # align the prediction to the reference part
            rot_alignment = transforms.quaternion_multiply(
                ref_part_gt_quat, transforms.quaternion_invert(ref_part_pred_quat)
            )
            trans_alignment = ref_part_gt_trans - transforms.quaternion_apply(
                rot_alignment, ref_part_pred_trans
            )
            # broadcast
            rot_alignment = rot_alignment.repeat_interleave(
                data_dict["num_parts"], dim=0
            )
            trans_alignment = trans_alignment.repeat_interleave(
                data_dict["num_parts"], dim=0
            )
            pred_trans = (
                transforms.quaternion_apply(rot_alignment, pred_trans) + trans_alignment
            )
            pred_rots = transforms.quaternion_multiply(rot_alignment, pred_rots)

        # Recover SE3 back to padded mode
        pred_trans_padded = torch.zeros(
            (B, P, 3), device=pred_trans.device, dtype=pred_trans.dtype
        )
        pred_rots_padded = torch.zeros(
            (B, P, 4), device=pred_rots.device, dtype=pred_rots.dtype
        )
        gt_trans_padded = torch.zeros(
            (B, P, 3), device=gt_trans.device, dtype=pred_trans.dtype
        )
        gt_rots_padded = torch.zeros(
            (B, P, 4), device=gt_rots.device, dtype=pred_rots.dtype
        )
        pred_trans_padded[part_valids] = pred_trans
        pred_rots_padded[part_valids] = pred_rots
        gt_trans_padded[part_valids] = gt_trans.to(dtype=gt_trans_padded.dtype)
        gt_rots_padded[part_valids] = gt_rots.to(dtype=gt_rots_padded.dtype)

        # When calculating the metrics, we should take care of the redundant parts.
        # The logic here is that, we only consider the valid parts for the metrics calculation.
        # i.e. Only measure how much will the result be affacted by the redundant parts.
        num_parts_wo_redundancy = (
            data_dict["num_parts"] - data_dict["redundancy"]
        )  # (B,)
        # to B, P like part_valids
        part_valids_wo_redundancy = (
            torch.cumsum(part_valids, dim=-1) <= num_parts_wo_redundancy[:, None]
        ) & part_valids

        # Two scenarios: (B, P, N, 3) or (B, N_sum, 3)
        # First one is for uniform sampling, second one is for weighted sampling
        # We have to calculate shape_cd and part_acc differently

        # Uniform sampling
        if pts.ndim == 4:
            B, P, N, C = pts.shape
            # (B, P, N, 1)
            expanded_part_scale = data_dict["scale"].unsqueeze(-1).expand(-1, -1, N, -1)
            pts = pts * expanded_part_scale  # (B, P, N, 3)

            acc, _, _ = calc_part_acc(
                pts,
                trans1=pred_trans_padded,
                trans2=gt_trans_padded,
                rot1=pred_rots_padded,
                rot2=gt_rots_padded,
                valids=part_valids_wo_redundancy,
            )

            shape_cd = calc_shape_cd(
                pts,
                trans1=pred_trans_padded,
                trans2=gt_trans_padded,
                rot1=pred_rots_padded,
                rot2=gt_rots_padded,
                valids=part_valids_wo_redundancy,
            )
        else:
            B, N_sum, C = pts.shape
            scale = data_dict["scale"][part_valids]
            scale = scale.repeat_interleave(points_per_part[part_valids], dim=0)
            pts = (pts.view(-1, C) * scale).view(B, N_sum, C)

            # Calculate Part Acc
            acc = calc_part_acc_weighted(
                pts,
                gt_trans=gt_trans,
                gt_rots=gt_rots,
                pred_trans=pred_trans,
                pred_rots=pred_rots,
                points_per_part=points_per_part,
                part_valids=part_valids,
                part_valids_wo_redundancy=part_valids_wo_redundancy,
            )

            # Calculate Shape Chamfer Distance
            shape_cd = calc_shape_cd_weighted(
                pts,
                gt_trans=gt_trans,
                gt_rots=gt_rots,
                pred_trans=pred_trans,
                pred_rots=pred_rots,
                points_per_part=points_per_part,
                part_valids=part_valids,
                part_valids_wo_redundancy=part_valids_wo_redundancy,
            )

        rmse_r = rot_metrics(
            pred_rots_padded, gt_rots_padded, part_valids_wo_redundancy, "rmse"
        )
        rmse_t = trans_metrics(
            pred_trans_padded, gt_trans_padded, part_valids_wo_redundancy, "rmse"
        )

        self.acc_list.append(acc)
        self.rmse_r_list.append(rmse_r)
        self.rmse_t_list.append(rmse_t)
        self.cd_list.append(shape_cd)

        if self.inference_config.get("write_to_json", True):
            save_dir = os.path.join(self.trainer.log_dir, "json_results")
            os.makedirs(
                save_dir,
                exist_ok=True,
            )
            for b in range(B):
                data = {
                    "name": data_dict["name"][b],
                    "num_parts": data_dict["num_parts"][b].detach().item(),
                    "gt_trans_rots": gt_trans_and_rots[
                        num_parts_cum[b] : num_parts_cum[b + 1]
                    ]
                    .detach()
                    .tolist(),
                    "pred_trans_rots": [
                        all_steps_preds[step_idx][
                            num_parts_cum[b] : num_parts_cum[b + 1]
                        ]
                        .detach()
                        .tolist()
                        for step_idx in range(len(all_steps_preds))
                    ],
                    "part_acc": acc[b].detach().item(),
                    "rmse_t": rmse_t[b].detach().item(),
                    "rmse_r": rmse_r[b].detach().item(),
                    "shape_cd": shape_cd[b].detach().item(),
                    "removal_pieces": data_dict["removal_pieces"][b],
                    "redundant_pieces": data_dict["redundant_pieces"][b],
                    "pieces": data_dict["pieces"][b],
                    "mesh_scale": data_dict["mesh_scale"][b].detach().item(),
                }

                json.dump(
                    data,
                    open(
                        os.path.join(
                            save_dir,
                            f"{data_dict['index'][b].item()}.json",
                        ),
                        "w",
                    ),
                )

        # save mesh results for visualization
        if self.inference_config.get("save_assembly", True) and "meshes" in data_dict:
            save_root = (
                Path(self.trainer.log_dir or self.trainer.default_root_dir or "./")
                / "assembly_results"
            )
            save_root.mkdir(parents=True, exist_ok=True)

            pred_trans_rots = torch.cat([pred_trans, pred_rots], dim=-1)

            for b in range(B):
                obj_dir = save_root / str(data_dict["name"][b])
                obj_dir.mkdir(parents=True, exist_ok=True)

                start = num_parts_cum[b].item()
                end = num_parts_cum[b + 1].item()

                scene_pred = trimesh.Scene()
                for part_idx, part_mesh in enumerate(data_dict["meshes"][b]):
                    global_idx = start + part_idx
                    gt_tf = gt_trans_and_rots[global_idx]
                    pred_tf = pred_trans_rots[global_idx]
                    gt_mat = self.se3_to_matrix(gt_tf)
                    pred_mat = self.se3_to_matrix(pred_tf)
                    T_final = pred_mat.cpu() @ torch.linalg.inv(gt_mat.cpu())
                    scene_pred.add_geometry(
                        part_mesh.copy(), transform=T_final.cpu().numpy()
                    )
                scene_pred.export(obj_dir / "view_assembly_0.glb")

                scene_gt = trimesh.Scene()
                for part_mesh in data_dict["meshes"][b]:
                    scene_gt.add_geometry(part_mesh)
                scene_gt.export(obj_dir / "view_gt.glb")

                removal_pieces = (
                    data_dict["removal_pieces"][b]
                    if "removal_pieces" in data_dict
                    else ""
                )
                redundant_pieces = (
                    data_dict["redundant_pieces"][b]
                    if "redundant_pieces" in data_dict
                    else ""
                )
                pieces = (
                    data_dict["pieces"][b] if "pieces" in data_dict else ""
                )

                assembly_json = {
                    "part_acc": acc[b].detach().item(),
                    "rmse_t": rmse_t[b].detach().item(),
                    "rmse_r": rmse_r[b].detach().item(),
                    "shape_cd": shape_cd[b].detach().item(),
                    "num_parts": int(data_dict["num_parts"][b].item()),
                    "gt_transform": gt_trans_and_rots[start:end]
                    .detach()
                    .cpu()
                    .tolist(),
                    "pred_transform": pred_trans_rots[start:end]
                    .detach()
                    .cpu()
                    .tolist(),
                    "removal_pieces": removal_pieces,
                    "redundant_pieces": redundant_pieces,
                    "pieces": pieces,
                    "mesh_scale": data_dict["mesh_scale"][b].detach().item(),
                }
                with (obj_dir / "view_assembly_0.json").open("w") as f:
                    json.dump(assembly_json, f, indent=2)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def on_validation_epoch_end(self):
        total_acc = torch.mean(torch.cat(self.acc_list))
        total_rmse_t = torch.mean(torch.cat(self.rmse_t_list))
        total_rmse_r = torch.mean(torch.cat(self.rmse_r_list))
        total_shape_cd = torch.mean(torch.cat(self.cd_list))

        self.log(f"eval/part_acc", total_acc, sync_dist=True)
        self.log(f"eval/rmse_t", total_rmse_t, sync_dist=True)
        self.log(f"eval/rmse_r", total_rmse_r, sync_dist=True)
        self.log(f"eval/shape_cd", total_shape_cd, sync_dist=True)
        self.acc_list = []
        self.rmse_t_list = []
        self.rmse_r_list = []
        self.cd_list = []
        return total_acc, total_rmse_t, total_rmse_r, total_shape_cd

    def configure_optimizers(self):
        optimizer = self.optimizer(
            self.parameters(),
        )

        if self.lr_scheduler is None:
            return {
                "optimizer": optimizer,
            }

        lr_scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def _loss(
        self, data_dict: dict, output_dict: dict
    ) -> Tuple[dict[str, torch.Tensor], set[str]]:
        raise NotImplementedError

    def forward(self, data_dict: dict):
        raise NotImplementedError
