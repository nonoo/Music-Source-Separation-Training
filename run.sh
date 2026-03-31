#!/bin/bash
self=`readlink "$0"`
if [ -z "$self" ]; then
	self=$0
fi
scriptname=`basename "$self"`
scriptdir=${self%$scriptname}

cd $scriptdir

.venv/bin/python inference.py \
	--model_type scnet \
	--config_path configs/config_musdb18_scnet_xl_more_wide_v5.yaml \
	--start_check_point results/model_scnet_ep_36_sdr_10.0891.ckpt "$@"
