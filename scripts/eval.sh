EXPERIMENT_NAME="everyday_vol_one_step_init"
DATA_ROOT="../breaking_bad_vol.hdf5"
DATA_CATEGORIES="['everyday']"
CHECKPOINT_PATH="output/GARF.ckpt"

HYDRA_FULL_ERROR=1 python eval.py \
    seed=42 \
    experiment=denoiser_flow_matching \
    experiment_name=$EVAL_NAME \
    loggers=csv \
    loggers.csv.save_dir=logs/GARF \
    trainer.num_nodes=1 \
    trainer.devices=[0] \
    data.data_root=$DATA_ROOT \
    data.categories=$DATA_CATEGORIES \
    data.batch_size=64 \
    ckpt_path=$CHECKPOINT_PATH \
    ++data.random_anchor=false \
    ++model.inference_config.one_step_init=true