#!/usr/bin/env bash

echo "!#/usr/bin/bash" > job_script.sh
echo "source $(pwd)/../bin/activate" >> job_script.sh
echo "python3 $(pwd)/comments_2_parquet_part1.py" >> job_script.sh

srun -p comdata -A comdata --nodes=1 --mem=120G --time=48:00:00 job_script.sh

start_spark_and_run.sh 1 $(pwd)/comments_2_parquet_part2.py
