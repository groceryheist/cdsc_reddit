#!/usr/bin/bash
start_spark_cluster.sh
singularity exec  /gscratch/comdata/users/nathante/cdsc_base.sif spark-submit --master spark://$(hostname).hyak.local:7077 lsi_similarities.py author --outfile=/gscratch/comdata/output//reddit_similarity/subreddit_comment_authors_10k_LSI.feather --topN=10000
singularity exec /gscratch/comdata/users/nathante/cdsc_base.sif stop-all.sh
