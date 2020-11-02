## this should be run manually since we don't have a nice way to wait on parallel_sql jobs

#!/usr/bin/env bash

./parse_submissions.sh

start_spark_and_run.sh 1 $(pwd)/submissions_2_parquet_part2.py


