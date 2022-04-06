#!/usr/bin/env bash

source ./bin/activate
python3 tf_comments.py gen_task_list

for job in $(seq 1 50); do sbatch checkpoint_parallelsql.sbatch; done;
