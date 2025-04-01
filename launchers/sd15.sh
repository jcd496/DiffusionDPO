export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch src/diffusion_dpo/trainer/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=1 \
  --gradient_accumulation_steps=16 \
  --max_train_steps=10 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=10 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/workspace/datasets/vision_language/pick_a_pic_v2/" \
  --checkpointing_steps 50 \
  --beta_dpo 5000 \
   --output_dir="/workspace/dev"

