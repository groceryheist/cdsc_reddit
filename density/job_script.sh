#!/usr/bin/bash
source ~/.bashrc
echo $(hostname)
start_spark_cluster.sh
spark-submit --verbose --master spark://$(hostname):43015 overlap_density.py authors --inpath=../../data/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather --outpath=../../data/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather --agg=pd.DataFrame.sum
stop-all.sh
