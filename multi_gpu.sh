# !/bin/bash

############################################
# Single Image Dehazing
############################################

declare -A pairs=( [A]=4 [B]=5 [C]=6 [D]=7 )

for i in "${!pairs[@]}"; do
{
  j=${pairs[$i]}
  echo CUDA=$j DATA=$i Processing...
  CUDA_VISIBLE_DEVICES=$j python instruct_all_in_one/lowlight_pipeline_16fp.py --use_bn --seed 20 --iter 2 \
  --result_dir results/test_lowlight_fp16 --text_dir ./data/LOLv1/prompt \
  --sample_dir ./data/LOLv1/sub_lol_$i > /dev/null
}&
done
wait

# for i in "${!pairs[@]}"; do
# {
#   j=${pairs[$i]}
#   echo CUDA=$j DATA=$i Processing...
#   CUDA_VISIBLE_DEVICES=$j python instruct_all_in_one/lowlight_pipeline_16fp.py --use_bn --seed 123 --iter 2 \
#   --result_dir results/test_lowlight_16fp_123_02 --text_dir /mnt/workspace/lhq/all-in-one/2024-ICML-TAO/test_samples/LOL_256x256/P \
#   --sample_dir /mnt/workspace/lhq/all-in-one/2024-ICML-TAO/test_samples/LOL_256x256/sub_lol_$i > /dev/null
# }&
# done
# wait



# HQ=test_samples/HSTS_256x256/results;
# GT=test_samples/HSTS_256x256/original;
# python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
# python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;

############################################
# Low-light Image Enhancement
############################################

#python instruct_all_in_one/colorizing_pipeline.py --use_bn --iter 0 --result_dir ./results/colorize/seed20_iter0
#python instruct_all_in_one/colorizing_pipeline.py --use_bn --iter 2 --result_dir ./results/colorize/seed123_iter2 --seed 123
############################################
# Single Image Denoising
############################################

# declare -A pairs=( [A]=4 [B]=5 [C]=6 [D]=7 )

# for i in "${!pairs[@]}"; do
# {
#   j=${pairs[$i]}
#   echo CUDA=$j DATA=$i Processing...
#   CUDA_VISIBLE_DEVICES=$j python sample_denoising.py \
#   --sample_dir test_samples/Kodak24_256x256/sigma30_$i > /dev/null
# }&
# done
# wait

# HQ=test_samples/Kodak24_256x256/results;
# GT=test_samples/Kodak24_256x256/original;
# python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
# python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;