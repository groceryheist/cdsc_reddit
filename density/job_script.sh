#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 overlap_density.py wang_overlaps --inpath=/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet --to_date=2020-04-13
stop-all.sh
