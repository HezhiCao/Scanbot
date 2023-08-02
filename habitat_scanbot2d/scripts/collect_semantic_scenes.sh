#!/bin/bash

set -e

if [[ $# -lt 3 ]]; then
    echo
    echo "Usage: collect_semantic_scenes.sh <src_glb_dir> <dest_dir> <semantic_dir>..."
    echo
    echo "**************************** Parameters ************************************"
    echo "src_glb_dir:  source directory that only contains .glb files"
    echo "dest_dir:     destination directory for aggregated .glb and semantic.ply files"
    echo "semantic_dir: one or more directories contains semantic.ply files"
    exit 1
fi

glb_dir=$1
dest_dir=$2
shift 2
semantic_dir=("$@")

if ! [[ -d $dest_dir ]]; then
    mkdir -p "$dest_dir"
fi

readarray glb_scenes < <(find "$glb_dir" -name "*.glb" | sort)
for glb_scene in "${glb_scenes[@]}"; do
    scene_name=$(basename $glb_scene .glb)
    semantic_ply=$(find "${semantic_dir[@]}" -name "${scene_name}_semantic.ply")
    if [[ -n $semantic_ply ]]; then
        cp -v "$glb_dir"/"$scene_name".* "$dest_dir"
        cp -v "$semantic_ply" "$dest_dir"
    fi
done
