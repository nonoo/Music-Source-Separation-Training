#!/bin/bash
self=`readlink "$0"`
if [ -z "$self" ]; then
	self=$0
fi
scriptname=`basename "$self"`
scriptdir=${self%$scriptname}

cd $scriptdir

.venv/bin/python3 ensemble.py --files $1 $2 --weights $3 $4 --type avg_wave --output $5
