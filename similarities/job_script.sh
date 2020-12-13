#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 weekly_cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_10000_weely.parquet
stop-all.sh
