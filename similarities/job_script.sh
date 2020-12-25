#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 wang_similarity.py --infile=/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet --max_df=10 --outfile=/gscratch/comdata/output/reddit_similarity/wang_similarity_1000_max10.feather
stop-all.sh
