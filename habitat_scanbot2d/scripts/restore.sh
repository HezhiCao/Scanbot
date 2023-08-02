#!/bin/bash

backup_no=0

for backup in data/new_checkpoints.backup*; do
    current_no="${backup#*new_checkpoints.backup}"
    backup_no=$((current_no > backup_no ? current_no : backup_no))
done

if [[ $backup_no -eq 0 ]]; then
    exit 1
fi

if ! [[ -e "data/new_checkpoints" ]]; then
    mv "data/new_checkpoints.backup$backup_no" data/new_checkpoints
fi

if ! [[ -e tb ]]; then
    mv "tb.backup$backup_no" tb
fi

if ! [[ -e train.log ]]; then
    mv "train.log.backup$backup_no" train.log
fi
