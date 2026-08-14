#!/bin/bash

set -e

scene='Museum'
ratio_threshold=1.1
lr_weight_maps_path=step1_${scene}/lr/${scene}/weight_maps   # reuse progressive Step 1's LR model (same r=16 --eval)
data_dir=data/tandt/${scene}_v2

splatsure_env=splatsure
stablesr_env=stablesr2
stablesr_dir=StableSR
stablesr_ckpt=stablesr_turbo.ckpt
vqgan_ckpt=vqgan_cfw_00011.ckpt
ddpm_steps=4

declare -A r_render_for_upscale=( [2]=8 [4]=4 [8]=2 [16]=1 )

run_stablesr () {
  local init_img=$1
  local outdir=$2
  local upscale=$3
  rm -rf "${outdir}"
  ( cd ${stablesr_dir} && \
    conda run -n ${stablesr_env} --no-capture-output \
    env PYTHONPATH=.:scripts:src/taming-transformers:src/clip \
    python scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py \
      --config configs/stableSRNew/v2-finetune_text_T_512.yaml \
      --ckpt ${stablesr_ckpt} --vqgan_ckpt ${vqgan_ckpt} \
      --init-img ../${init_img} --outdir ../${outdir} \
      --ddpm_steps ${ddpm_steps} --dec_w 0.5 --colorfix_type adain \
      --vqgantile_stride 500 --vqgantile_size 512 \
      --input_downsample 16 --upscale ${upscale} )

  ( cd ${outdir} && for f in *.png; do mv -- "$f" "${f%.png}.jpg"; done )
}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${splatsure_env}

for upscale in 2 4 8 16; do
  r_render=${r_render_for_upscale[$upscale]}
  sr_images_dir=${data_dir}/images_SR_up${upscale}
  output_dir=outputs_baseline_${scene}_up${upscale}/${scene}

  run_stablesr "${data_dir}/images" "${sr_images_dir}" ${upscale}

  PYTHONPATH=. python src/train.py -s ${data_dir} -m ${output_dir} -r 1 --eval \
    --images images_SR_up${upscale} --img_ext jpg --upscale ${upscale} \
    --weight_maps_path ${lr_weight_maps_path}

  PYTHONPATH=. python src/render.py --model_path ${output_dir} --images images \
    -r ${r_render} --img_ext jpg --upscale 1

  PYTHONPATH=. python src/metrics.py -m ${output_dir}/test/ours_30000
done

echo "Baseline comparison complete."