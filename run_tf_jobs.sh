#!/usr/bin/env bash
module load parallel_sql
source ../bin/activate
python3 tf_comments.py gen_task_list
psu --del --Y
cat tf_task_list | psu --load

for job in $(seq 1 50); do sbatch checkpoint_parallelsql.sbatch; done;
