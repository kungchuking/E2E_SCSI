#!/bin/csh

set arg_count = $#argv
if ( $arg_count >= 1 ) then
	if ( "$argv[1]" == "-clean" || "$argv[1]" == "-clean_only" ) then
		echo "[INFO] Killing alll other GPU processes to free up resources."
	
		sh -c 'ps | grep python | sed "s/ pts.\+$//g" > .tmp.csh'
		chmod +x .tmp.csh
		sed -i "s/^/kill -9 /g" .tmp.csh
		source .tmp.csh
		rm -rf .tmp.csh
		rm -rf debug_rank_*
		rm -rf dynamicstereo_sf_dr
	endif

	if ( "$argv[1]" == "-clean_only" ) then
		exit 0
	endif
endif

setenv PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:32,garbage_collection_threshold:0.5,expandable_segments:False"
setenv CUDA_LAUNCH_BLOCKING 1
setenv PYTORCH_NO_CUDA_MEMORY_CACHING 1
setenv CUBLAS_WORKSPACE_CONFIG ":16:8"

# -- GPU OOM Error when trained with sample_len=8 on kilby.
python -m rmm.alloc train.py --batch_size 1 \
  --spatial_scale -0.2 0.4 --image_size 480 640 --saturation_range 0 1.4 --num_steps 200000  \
  --ckpt_path dynamicstereo_sf_dr  \
  --sample_len 16 --lr 0.0003 --train_iters 8 --valid_iters 8    \
  --num_workers 28 --save_freq 100  --update_block_3d --different_update_blocks \
  --attention_type self_stereo_temporal_update_time_update_space --train_datasets dynamic_replica
