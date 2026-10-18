#!/bin/bash
# DreamZero KR1-sim Training Script (Kinisi KR1 sim cube-pick rig)
#
# Usage:
#   DATA_ROOT=/path/to/dreamzero_exports/<config>/<repo> bash scripts/train/kr1_sim_training.sh [extra hydra overrides]
#
# Prerequisites:
#   - KR1 dataset exported by mimic_dex's export_dreamzero.py and converted with
#     scripts/data/convert_lerobot_to_gear.py --embodiment-tag kr1_sim (the export's
#     gear_args.json carries the exact arguments).
#     state/action: 20-D bi-arm EE layout (right arm neutral-masked);
#     video: left_wrist_camera; annotation.task.
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded from HuggingFace)
#   - DreamZero-AgiBot pretrained checkpoint (for loading LoRA weights before fine-tuning)
#     git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot ./checkpoints/DreamZero-AgiBot

export HYDRA_FULL_ERROR=1

# ============ CHANGE THESE VARIABLES ============
KR1_SIM_DATA_ROOT=${DATA_ROOT:-"./data/kr1_sim_lerobot"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_kr1_sim_lora"}

# Number of GPUs to use (default: all visible GPUs)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# frame_seqlen is TOKENS PER COMPOSITED LATENT FRAME: (H/16)x(W/16) per 320x176
# tile = 220; multi-view recipes tile a 2x2 grid (880). One untiled view
# = 220 (the local collate shows (9, 176, 320, 3) image stacks).
NUM_VIEWS=${NUM_VIEWS:-1}
FRAME_SEQLEN=${FRAME_SEQLEN:-220}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$KR1_SIM_DATA_ROOT" ] || [ ! -f "$KR1_SIM_DATA_ROOT/meta/embodiment.json" ]; then
    echo "ERROR: KR1 GEAR dataset not found at $KR1_SIM_DATA_ROOT"
    echo "Export it with mimic_dex's export_dreamzero.py, then run"
    echo "scripts/data/convert_lerobot_to_gear.py with the export's gear_args.json."
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/kr1_sim \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=9 \
    action_horizon=24 \
    num_views=$NUM_VIEWS \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=4 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=1 \
    frame_seqlen=$FRAME_SEQLEN \
    save_strategy=steps \
    kr1_sim_data_root=$KR1_SIM_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true \
    "$@"
