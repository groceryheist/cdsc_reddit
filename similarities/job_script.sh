#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet
stop-all.sh
