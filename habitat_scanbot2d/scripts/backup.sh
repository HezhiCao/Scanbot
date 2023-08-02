#!/bin/bash

backup_no=0

for backup in data/new_checkpoints.backup*; do
    current_no="${backup#*new_checkpoints.backup}"
    backup_no=$((current_no > backup_no ? current_no : backup_no))
done

(( ++backup_no ))

mv data/new_checkpoints "data/new_checkpoints.backup$backup_no"
mv tb "tb.backup$backup_no"
mv train.log "train.log.backup$backup_no"
