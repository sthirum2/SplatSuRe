#!/bin/bash

set -e

scene=${1:-Museum}
ratio_threshold=1.1
weight_maps_dirname=weight_maps
upscale=2
n_steps=4

splatsure_env=splatsure
stablesr_env=stablesr2
stablesr_dir=StableSR
stablesr_ckpt=stablesr_turbo.ckpt
vqgan_ckpt=vqgan_cfw_00011.ckpt
ddpm_steps=4

if [[ $scene = @(Auditorium|Ignatius|Palace|Ballroom|Courthouse|Panther|Barn|Lighthouse|Playground|Courtroom|M60|Temple|Caterpillar|Family|Meetingroom|Train|Francis|Truck|Church|Horse|Museum) ]]; then
  data_dir=data/tandt/${scene}_v2
fi
if [[ $scene = @(bicycle|bonsai|counter|flowers|garden|kitchen|room|stump|treehill) ]]; then
  data_dir=data/mipnerf_data/${scene}_v2
fi
if [[ $scene = @(drjohnson|playroom) ]]; then
  data_dir=data/deep_blending/${scene}_v2/colmap
fi

# Per-step downsample factor for train_lr/weight_maps (r), and render resolution (r_render)
r_values=(16 1 1 1)
r_render_values=(8 4 2 1)

run_stablesr () {
  local init_img=$1
  local outdir=$2
  local input_downsample=$3
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
      --input_downsample ${input_downsample} --upscale ${upscale} )

  # StableSR writes .png; rename to .jpg so train_lr.py / weight_maps.py
  # (which have no --img_ext flag) resolve the correct extension.
  ( cd ${outdir} && for f in *.png; do mv -- "$f" "${f%.png}.jpg"; done )
}

rebuild_step_images () {
  local render_root=$1
  local out_dir=$2
  mkdir -p "${out_dir}"
  cp "${render_root}/train/ours_30000/renders/"*.png "${out_dir}/" 2>/dev/null
  cp "${render_root}/test/ours_30000/renders/"*.png "${out_dir}/" 2>/dev/null
}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${splatsure_env}

for step in $(seq 1 ${n_steps}); do
  echo "=================== STEP ${step} ==================="
  idx=$((step - 1))
  r=${r_values[$idx]}
  r_render=${r_render_values[$idx]}
  output_dir=step${step}_${scene}
  lr_model=${output_dir}/lr/${scene}
  sr_model=${output_dir}/${scene}

  if [[ $step -eq 1 ]]; then
    prev_render_images=images        # raw photos
    input_downsample=${r}            # StableSR downsamples raw images itself
  else
    prev_step_images=images_step$((step - 1))_sr
    input_downsample=1               # already at the right scale, just 2x upscale
  fi
  sr_images_dir=images_step$((step - 1))_StableSR
  [[ $step -eq 1 ]] && sr_images_dir=images_16_2x

  # Rebuild this step's input from the previous step's render (steps 2+) 
  if [[ $step -gt 1 ]]; then
    rebuild_step_images "step$((step - 1))_${scene}/${scene}" "${data_dir}/${prev_step_images}"
  fi

  # StableSR 
  if [[ $step -eq 1 ]]; then
    run_stablesr "${data_dir}/images" "${data_dir}/${sr_images_dir}" ${r}
  else
    run_stablesr "${data_dir}/${prev_step_images}" "${data_dir}/${sr_images_dir}" 1
  fi

  # Train LR model + weight maps (skip for step 1's raw-image LR, which uses -r; steps 2+ use the previous StableSR images at native res) ---
  PYTHONPATH=. python src/train_lr.py -s ${data_dir} -m ${lr_model} -r ${r} --eval --skip_test $( [[ $step -gt 1 ]] && echo "--images ${sr_images_dir} --img_ext jpg" )

  PYTHONPATH=. python src/weight_maps.py -s ${data_dir} -m ${lr_model} -r ${r} --eval --weight_maps_dirname ${weight_maps_dirname} --ratio_threshold ${ratio_threshold} $( [[ $step -gt 1 ]] && echo "--images ${sr_images_dir}" )

  # Train SR model                 
  PYTHONPATH=. python src/train.py -s ${data_dir} -m ${sr_model} -r 1 --eval --skip_test --images ${sr_images_dir} --img_ext jpg --upscale ${upscale} --weight_maps_path ${lr_model}/${weight_maps_dirname}

  # Render at this step's target resolution ---
  # render.py's own test/ours_30000/{renders,gt} pair already compares against
  # the TRUE original photo (downsampled to this step's native resolution) --
  # confirmed via pixel diff (0.0 vs real photo, ~22 vs StableSR training input).
  # No separate render_original_test_split.py / rebuilt-gt step needed.
  PYTHONPATH=. python src/render.py --model_path ${sr_model} --images images -r ${r_render} --img_ext jpg --upscale 1

  # Metrics: compare stepN's test renders against the real photo, same resolution 
  PYTHONPATH=. python src/metrics.py -m ${sr_model}/test/ours_30000
done

echo "Progressive pipeline complete: steps 1-${n_steps} for scene ${scene}."

