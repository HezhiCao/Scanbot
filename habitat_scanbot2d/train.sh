#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
python -u -m torch.distributed.run \
    --nproc_per_node 8 \
    global_training.py \
