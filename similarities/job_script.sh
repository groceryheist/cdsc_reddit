#!/usr/bin/bash
start_spark_cluster.sh
singularity exec  /gscratch/comdata/users/nathante/cdsc_base.sif spark-submit --master spark://$(hostname):7077 top_subreddits_by_comments.py 
singularity exec /gscratch/comdata/users/nathante/cdsc_base.sif stop-all.sh
