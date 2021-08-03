#!/usr/bin/bash
start_spark_cluster.sh
singularity exec  /gscratch/comdata/users/nathante/cdsc_base.sif spark-submit --master spark://$(hostname).hyak.local:7077 overlap_density.py authors --inpath=/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather --outpath=/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather --agg=pd.DataFrame.sum
singularity exec /gscratch/comdata/users/nathante/cdsc_base.sif stop-all.sh
