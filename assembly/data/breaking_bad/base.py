from typing import Literal, List

import trimesh
import logging
import h5py
import numpy as np
from torch.utils.data import Dataset, default_collate

COLORS = [
    [254, 138, 24],
    [201, 26, 9],
    [35, 120, 65],
    [0, 85, 191],
    [242, 112, 94],
    [252, 151, 172],
    [75, 159, 74],
    [0, 143, 155],
    [245, 205, 47],
    [67, 84, 163],
    [179, 215, 209],
    [199, 210, 60],
    [255, 128, 13],
]


class BreakingBadBase(Dataset):
    """Base Dataset for the Breaking Bad dataset."""

    COLORS = COLORS

    def __init__(
        self,
        split: Literal["train", "val", "test"] = "train",
        data_root: str = "data",
        category: Literal["everyday", "artifact", "other", "all"] = "everyday",
        min_parts: int = 2,
        max_parts: int = 20,
        num_points_to_sample: int = 8192,
        min_points_per_part: int = 20,
        num_removal: int = 0,
        num_redundancy: int = 0,
        multi_ref: bool = False,
        mesh_sample_strategy: Literal["uniform", "poisson"] = "uniform",
        random_anchor: bool = False,
    ):
        super().__init__()
        self.split = split
        self.data_root = data_root
        self.category = category
        self.min_parts = min_parts
        self.max_parts = max_parts
        self.num_points_to_sample = num_points_to_sample
        self.min_points_per_part = min_points_per_part
        self.num_removal = num_removal
        self.num_redundancy = num_redundancy
        self.multi_ref = multi_ref
        self.mesh_sample_strategy = mesh_sample_strategy
        self.random_anchor = random_anchor
        self.data_list = self.get_data_list()

        print("Using mesh sample strategy:", self.mesh_sample_strategy)
        trimesh.util.log.setLevel(logging.ERROR)
        # Cannot enable removal and redundancy at the same time
        assert not (self.num_removal and self.num_redundancy), (
            "Cannot enable removal and redundancy at the same time"
        )

    def get_data_list(self) -> List[str]:
        """Return the list of data samples."""
        h5_file = h5py.File(self.data_root, "r")
        if self.category == "all":
            everyday_objs = list(h5_file["data_split"]["everyday"][self.split])
            artifact_objs = list(h5_file["data_split"]["artifact"][self.split])
            data_list = everyday_objs + artifact_objs
        else:
            data_list = list(h5_file["data_split"][self.category][self.split])
        data_list = [d.decode("utf-8") for d in data_list]
        filtered_data_list = []
        for item in data_list:
            try:
                num_parts = len(h5_file[item]["pieces"].keys())
                # Here's the limit
                # For removal, we need to ensure that, after removal, number of parts should still greater than min_parts
                # For redundancy, we need to ensure that,
                # 1. after redundancy, number of parts should still less than max_parts
                # 2. num of redundancy should not exceed num of parts
                if (
                    self.min_parts + self.num_removal
                    <= num_parts
                    <= self.max_parts - self.num_redundancy
                    and num_parts > self.num_redundancy
                ):
                    filtered_data_list.append(item)
            except:
                continue

        h5_file.close()
        return filtered_data_list

    def get_meshes(self, name: str) -> List[trimesh.Trimesh]:
        h5_file = h5py.File(self.data_root, "r")
        pieces = h5_file[name]["pieces"].keys()

        meshes = [
            {
                "vertices": np.array(h5_file[name]["pieces"][piece]["vertices"][:]),
                "faces": np.array(h5_file[name]["pieces"][piece]["faces"][:]),
                "color": self.COLORS[idx % len(self.COLORS)],
            }
            for idx, piece in enumerate(pieces)
        ]
        h5_file.close()
        return meshes

    def get_data(self, index: int):
        name = self.data_list[index]

        h5_file = h5py.File(self.data_root, "r")
        pieces = h5_file[name]["pieces"].keys()
        pieces_names = h5_file[name]["pieces_names"][:]
        pieces_names = [name.decode("utf-8") for name in pieces_names]
        num_parts = len(pieces)
        meshes = [
            trimesh.Trimesh(
                vertices=np.array(h5_file[name]["pieces"][piece]["vertices"][:]),
                faces=np.array(h5_file[name]["pieces"][piece]["faces"][:]),
            )
            for piece in pieces
        ]
        meshes_max_scale = 1.0
        for i in range(num_parts):
            extents = meshes[i].extents
            meshes_max_scale = max(meshes_max_scale, max(extents))
        meshes = [mesh.apply_scale(1.0 / meshes_max_scale) for mesh in meshes]

        shared_faces = [
            (
                np.array(h5_file[name]["pieces"][piece]["shared_faces"][:])
                if "shared_faces" in h5_file[name]["pieces"][piece]
                else -np.ones(len(meshes[idx].faces), dtype=np.int64)
            )
            for idx, piece in enumerate(pieces)
        ]

        graph = self.get_graph(shared_faces=shared_faces)

        # Removal to simulate missing part
        removal_pieces = []
        if self.num_removal > 0:
            num_parts -= self.num_removal
            removal_mask = h5_file[name]["removal_masks"][self.num_removal - 1]
            meshes = np.array(meshes)[removal_mask]
            shared_faces = np.array(shared_faces, dtype="object")[removal_mask]

            removal_order = np.array(h5_file[name]["removal_order"][: self.num_removal])
            removal_pieces = list(
                np.array(
                    [name.decode("utf-8") for name in h5_file[name]["pieces_names"]]
                )[removal_order]
            )
            assert len(meshes) == num_parts
            assert len(shared_faces) == num_parts

        # Redundancy to simulate extra part
        redundant_pieces = []
        if self.num_redundancy > 0:
            assert num_parts > self.num_redundancy, (
                "num of parts should greater than num of redundancy"
            )
            redundant_pieces = h5_file[name]["redundant_pieces"][: self.num_redundancy]
            redundant_pieces = [
                f"{p[0].decode('utf-8')}/pieces/{p[1].decode('utf-8')}"
                for p in redundant_pieces
            ]
            redundant_meshes = [
                trimesh.Trimesh(
                    vertices=np.array(h5_file[p]["vertices"][:]),
                    faces=np.array(h5_file[p]["faces"][:]),
                )
                for p in redundant_pieces
            ]
            redundant_shared_faces = [
                -np.ones(len(mesh.faces), dtype=np.int64) for mesh in redundant_meshes
            ]

            # Extend the original meshes and shared_faces
            # No need to update the graph, since we already padded to max_parts,
            # and the redundant parts are not connected to any other parts
            meshes.extend(redundant_meshes)
            shared_faces.extend(redundant_shared_faces)
            num_parts += self.num_redundancy

        h5_file.close()

        pointclouds_gt, pointclouds_normals_gt, fracture_surface_gt = (
            self.sample_points(
                meshes=meshes,
                shared_faces=shared_faces,
            )
        )

        data = {
            "index": index,
            "name": name,
            "num_parts": num_parts,
            "pointclouds_gt": pointclouds_gt,
            "pointclouds_normals_gt": pointclouds_normals_gt,
            "fracture_surface_gt": fracture_surface_gt,
            "graph": graph,
            "removal_pieces": ",".join(removal_pieces),
            "redundant_pieces": ",".join(redundant_pieces),
            "pieces": ",".join(pieces_names),
            "mesh_scale": meshes_max_scale,
            "meshes": meshes,
        }

        # For training, pickle the meshes will take too much spaces and time and is not needed.
        if self.split == "train":
            del data["meshes"]

        return data

    def transform(self, data: dict):
        raise NotImplementedError

    def sample_points(
        self,
        meshes: List[trimesh.Trimesh],
        shared_faces: List[np.ndarray],
    ) -> List[np.ndarray]:
        raise NotImplementedError

    def get_graph(
        self,
        shared_faces: List[np.ndarray],
    ) -> np.ndarray:
        """
        Get the connectivity matrix of a list of meshes.

        Args:
            shared_faces: List of shared faces of meshes.

        Returns:
            np.ndarray: Graph matrix.
        """
        num_parts = len(shared_faces)
        graph = np.zeros((self.max_parts, self.max_parts), dtype=bool)
        parts_indices = np.arange(num_parts)

        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                if j in shared_faces[i]:
                    graph[i, j] = graph[j, i] = 1

        return graph.astype(bool)

    def _pad_data(self, input_data: np.ndarray):
        """Pad data to shape [`self.max_parts`, data.shape[1], ...]."""
        d = np.array(input_data)
        pad_shape = (self.max_parts,) + tuple(d.shape[1:])
        pad_data = np.zeros(pad_shape, dtype=np.float32)
        pad_data[: d.shape[0]] = d
        return pad_data

    def __getitem__(self, index):
        data = self.get_data(index)
        data = self.transform(data)
        return data

    def visualize(self, index: int):
        raise NotImplementedError

    def __len__(self):
        return len(self.data_list)

    def export_hdf5(self, output_path: str):
        # Using dataloder for easy multi-processing
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor

        f = h5py.File(output_path, "w")
        # Write Metadata
        f.attrs["dataset"] = self.__class__.__name__
        f.attrs["split"] = self.split
        f.attrs["data_root"] = self.data_root
        f.attrs["category"] = self.category
        f.attrs["min_parts"] = self.min_parts
        f.attrs["max_parts"] = self.max_parts
        f.attrs["num_points_to_sample"] = self.num_points_to_sample
        f.attrs["min_points_per_part"] = self.min_points_per_part
        f.attrs["num_samples"] = len(self)

        pool = ProcessPoolExecutor(max_workers=48)
        for data in tqdm(
            pool.map(
                self.get_data,
                range(len(self)),
            ),
            desc="Exporting data",
            total=len(self),
        ):
            name = data["name"]
            num_parts = data["num_parts"]
            pointclouds_gt = data["pointclouds_gt"]
            pointclouds_normals_gt = data["pointclouds_normals_gt"]
            fracture_surface_gt = data["fracture_surface_gt"]
            graph = data["graph"]

            group = f.create_group(name)
            group.create_dataset("num_parts", data=num_parts)
            group.create_dataset("pointclouds_gt", data=pointclouds_gt)
            group.create_dataset("pointclouds_normals_gt", data=pointclouds_normals_gt)
            group.create_dataset("fracture_surface_gt", data=fracture_surface_gt)
            group.create_dataset("graph", data=graph)

        f.close()

    @staticmethod
    def collate_fn(batch):
        collated_batch = {}
        for key in batch[0].keys():
            if key == "meshes":
                collated_batch[key] = [item[key] for item in batch]
            else:
                collated_batch[key] = default_collate([item[key] for item in batch])

        return collated_batch
