#!/usr/bin/bash
start_spark_cluster.sh
spark-submit --master spark://$(hostname):18899 overlap_density.py authors --inpath=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather --outpath=/gscratch/comdata/output/reddit_density/comment_authors_10000.feather --agg=pd.DataFrame.sum
stop-all.sh
