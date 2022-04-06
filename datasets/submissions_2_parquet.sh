#!/usr/bin/env bash
## this should be run manually since we don't have a nice way to wait on parallel_sql jobs


srun -p compute-bigmem -A comdata --nodes=1 --mem-per-cpu=9g -c 40 --time=120:00:00 python3 $(pwd)/submissions_2_parquet_part1.py gen_task_list

start_spark_and_run.sh 1 $(pwd)/submissions_2_parquet_part2.py


