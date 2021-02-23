#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather
stop-all.sh
