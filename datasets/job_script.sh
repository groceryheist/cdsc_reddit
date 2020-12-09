#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 weekly_cosine_similarities.py term --outfile=/gscratch/comdata/users/nathante/subreddit_term_similarity_weekly_5000.parquet --topN=5000
stop-all.sh
