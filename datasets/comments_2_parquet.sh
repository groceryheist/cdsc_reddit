#!/usr/bin/env bash
## needs to be run by hand since i don't have a nice way of waiting on a parallel-sql job to complete 

echo "#!/usr/bin/bash" > job_script.sh
#echo "source $(pwd)/../bin/activate" >> job_script.sh
echo "python3 $(pwd)/comments_2_parquet_part1.py" >> job_script.sh

srun -p compute-bigmem -A comdata --nodes=1 --mem-per-cpu=9g -c 40 --time=120:00:00 --pty job_script.sh

start_spark_and_run.sh 1 $(pwd)/comments_2_parquet_part2.py
