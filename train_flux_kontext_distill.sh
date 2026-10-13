#!/bin/bash

# FLUX.1 Kontext 12B → FLUX.2 Klein 4B distillation training
#
# Prerequisites:
#   1. Cache latents + TE outputs for BOTH architectures on the SAME dataset:
#        python flux_kontext_cache_latents.py          --dataset_config datasets/distill_dataset/dataset.toml
#        python flux_kontext_cache_text_encoder_outputs.py --dataset_config datasets/distill_dataset/dataset.toml
#        python flux_2_cache_latents.py    --model_version klein-4b --dataset_config datasets/distill_dataset/dataset.toml
#        python flux_2_cache_text_encoder_outputs.py --model_version klein-4b --dataset_config datasets/distill_dataset/dataset.toml
#   2. All resolutions in the dataset TOML must be multiples of 32 (LCM of
#      Kontext ×16 and FLUX.2 ×32 bucketing) so both architectures produce
#      identical bucket resolutions and cache basenames.

# export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH
# module load cuda/12.4

pwd

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 --main_process_port 29500 \
    flux_kontext_distill_train.py \
    --sdpa \
    --mixed_precision bf16 \
    --gradient_checkpointing \
    \
    `# ---- Student (FLUX.2 Klein 4B) ----` \
    --dit         ComfyUI/models/diffusion_models/flux-2-klein-4b.safetensors \
    --model_version klein-4b \
    --vae         ComfyUI/models/vae/flux2_ae.safetensors \
    \
    `# ---- Teacher (FLUX.1 Kontext 12B) ----` \
    --distill_teacher_dit   ComfyUI/models/diffusion_models/flux1-kontext-dev.safetensors \
    --distill_teacher_dtype bf16 \
    --distill_teacher_vae   ComfyUI/models/vae/ae.safetensors \
    --distill_student_vae   ComfyUI/models/vae/flux2_ae.safetensors \
    --distill_guidance      2.5 \
    --distill_teacher_steps 15 \
    --distill_student_steps 4 \
    --distill_chunk_batches 50 \
    --distill_loss_type     mse \
    --distill_seed_mode     per_item \
    \
    `# ---- Dataset ----` \
    --dataset_config  datasets/distill_dataset/dataset.toml \
    \
    `# ---- LoRA / Network ----` \
    --network_module  networks.lora_flux_2 \
    --network_dim     32 \
    --network_alpha   32 \
    \
    `# ---- Optimizer / LR ----` \
    --optimizer_type adamw \
    --learning_rate 5e-5 \
    --optimizer_args "weight_decay=0.01" \
    --lr_scheduler constant_with_warmup \
    --lr_warmup_steps 50 \
    --timestep_sampling flux2_shift \
    --weighting_scheme "none" \
    \
    `# ---- Training ----` \
    --max_train_steps     5000 \
    --save_every_n_steps  500 \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    \
    `# ---- Output ----` \
    --output_dir  output/flux_kontext_distill \
    --output_name flux_kontext_to_klein_4b \
    --logging_dir output/flux_kontext_distill/logs \
    --log_prefix  distill \
    --save_state --save_state_on_train_end
    # --distill_teacher_dtype fp8_e4m3fn  # use fp8 teacher to halve VRAM (~12 GB vs ~24 GB)
    # --sample_every_n_steps 500 --sample_prompts datasets/distill_dataset/sample_prompts.json
