#!/usr/bin/env bash

# part2 should be run on one ore more spark nodes

./submissions_2_parquet_part1.py

start_spark_and_run.sh 1 $(pwd)/submissions_2_parquet_part2.py


