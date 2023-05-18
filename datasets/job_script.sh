#!/usr/bin/bash
source ~/.bashrc
echo $(hostname)
start_spark_cluster.sh
spark-submit --verbose --master spark://$(hostname):43015 submissions_2_parquet_part2.py 
stop-all.sh
