#!/usr/bin/bash
start_spark_cluster.sh
singularity exec  /gscratch/comdata/users/nathante/containers/nathante.sif spark-submit --master spark://$(hostname):7077 comments_2_parquet_part2.py 
singularity exec /gscratch/comdata/users/nathante/containers/nathante.sif stop-all.sh
