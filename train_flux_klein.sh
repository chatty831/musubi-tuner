
#!bin/bash

# export LD_LIBRARY_PATH=$CONDA_PREFIX/cuda-compat:$LD_LIBRARY_PATH

module load cuda/12.4

pwd

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 --main_process_port 29500 \
    musubi-tuner/flux_2_train_network.py \
    --compile \
    --compile_backend inductor \
    --compile_mode default \
    --compile_dynamic "true" \
    --cuda_allow_tf32 \
    --cuda_cudnn_benchmark \
    --compile_cache_size_limit 32 \
    --dit ComfyUI/models/diffusion_models/flux-2-klein-9b.safetensors \
    --model_version klein-base-9b \
    --vae ComfyUI/models/vae/ae.safetensors \
    --text_encoder ComfyUI/models/text_encoders/text_encoder/model-00001-of-00004.safetensors \
    --dataset_config musubi-tuner/datasets/bw_color_flux_dataset/dataset.toml \
    --xformers --mixed_precision bf16 \
    --lr_scheduler constant_with_warmup --lr_warmup_steps 30 \
    --timestep_sampling flux2_shift --weighting_scheme "none" \
    --optimizer_type adamw --learning_rate 5e-5 --gradient_checkpointing \
    --optimizer_args "weight_decay=0.01" \
    --max_data_loader_n_workers 3 --persistent_data_loader_workers \
    --network_module networks.lora_flux_2 \
    --network_dim 32 \
    --network_alpha 32 \
    --logging_dir musubi-tuner/output/flux2_lora_colorizer/logs \
    --log_prefix flux2_lora_colorizer \
    --max_train_steps 8000 --save_every_n_steps 500 \
    --sample_every_n_steps 500 --sample_prompts musubi-tuner/datasets/bw_color_flux_dataset/sample_prompts.json \
    --output_dir musubi-tuner/output/flux2_lora_colorizer --output_name flux2_lora_colorizer \
    --save_state --save_state_on_train_end \
    --ddp_gradient_as_bucket_view
    # --gradient_accumulation_steps 2 \
    # --network_dropout 0.05 \
    # --merge_lora_path ComfyUI/models/loras/converted_Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors \
    # --merge_lora_ratio 0.03 --warmup_dataset_config musubi-tuner/datasets/dataset.toml --post_merge_warmup_steps 50 \
    # --logit_mean -2.0 --logit_std 1.8 \
    # --guidance_scale 4.0 \
    # --fp8_vl \
    # --split_attn \
    # --lora_path ComfyUI/models/loras/lora_definer_qshift_v8-step00012000.safetensors \
    # --dit ComfyUI/models/diffusion_models/qwen_image_bf16.safetensors \
    # --weighting_scheme "sigma_sqrt" \
    # --network_args "loraplus_lr_ratio=4" \



# tensorboard --logdir=logs