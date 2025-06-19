#!/bin/bash

python generate_data.py --run_name=verification_run \
                        --dataset=./prompts/verification_prompts.csv \
                        --gen_with_wm \
                        --w_pattern=ring \
                        --w_channel=3 \
                        --num_inference_steps=2

base_dir="./logs"
log_dirs=($(find "$base_dir" -maxdepth 1 -mindepth 1 -type d -name "run-*-verification_run-*"))
data_dir=${log_dirs[0]}

python recover_latents.py $data_dir/media/wm_img/ \
                          $data_dir/media/wm_sd_latent/ \
                          data \
                          --model_id=stabilityai/stable-diffusion-2-1-base

python recover_latents.py $data_dir/media/no_wm_img/ \
                          $data_dir/media/no_wm_sd_latent/ \
                          data \
                          --model_id=stabilityai/stable-diffusion-2-1-base

python train_surrogate.py $data_dir/media/wm_sd_latent/data.pt \
                       $data_dir/media/no_wm_sd_latent/data.pt \
                       ./output/models/ \
                       verification_model \
                       --mode=latent \
                       --apply_fft \
                       --batch_size=2 \
                       --epochs=1

python remove_watermark.py $data_dir/media/wm_sd_latent/data.pt \
                           ./output/images/verification_set/ \
                           --model_save_path=./output/models/verification_model.pth \
                           --mode=latent \
                           --batch_size=2 \
                           --init_steps=1 \
                           --n_steps=1 \
                           --strength=32 \
                           --vae=stabilityai/stable-diffusion-2-1-base \
                           --apply_fft

python assess_images.py --run_name=verification_assess \
                        --original_images_path=$data_dir/media/wm_img/ \
                        --adv_images_path=./output/images/verification_set/ \
                        --table_path=$data_dir/media/table/metadata.csv \
                        --imagenet_path=./verification_imagenet/ \
                        --watermark_path=$data_dir/tr_params.pth \
                        --test_num_inference_steps=2 \
                        --reference_model=ViT-g-14 \
                        --reference_model_pretrain=laion2b_s12b_b42k