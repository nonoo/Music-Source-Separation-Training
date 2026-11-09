#!/bin/bash
self=`readlink "$0"`
if [ -z "$self" ]; then
	self=$0
fi
scriptname=`basename "$self"`
scriptdir=${self%$scriptname}

cd $scriptdir

.venv/bin/python inference.py \
	--model_type sub_roformer \
	--config_path configs/subroformer_base_4stem.yaml \
	--start_check_point results/sub_roformer_base_4stem.ckpt "$@"
