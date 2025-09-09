#!/bin/bash
data_path=tinynav_db
output_path=tinynav_db
MAX_JOBS=1 ns-train splatfacto \
      --output-dir $output_path \
      --experiment-name experiment \
      --method-name splatfacto \
      --timestamp 0 \
      --pipeline.model.cull_alpha_thresh=0.005 \
      --pipeline.model.use_scale_regularization True \
      --viewer.quit-on-train-completion True \
    nerfstudio-data \
      --data $data_path \
      --center-method none \
      --auto-scale-poses False \
      --orientation_method none
ns-export gaussian-splat --load-config "$output_path/experiment/splatfacto/0/config.yml"  --output-dir "$output_path"
