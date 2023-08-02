nsys profile --sample=none --trace=cuda,nvtx --trace-fork-before-exec=true --output=$1 python global_training.py
