#!/usr/bin/bash
source ~/.bashrc
echo $(hostname)
start_spark_cluster.sh
spark-submit --verbose --master spark://$(hostname):43015 tfidf.py authors --topN=100000 --inpath=../../data/reddit_ngrams/comment_authors_sorted.parquet --outpath=../../data/reddit_similarity/tfidf/comment_authors_100k.parquet
stop-all.sh
