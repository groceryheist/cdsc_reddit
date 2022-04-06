#!/usr/bin/bash
start_spark_cluster.sh
singularity exec  /gscratch/comdata/users/nathante/containers/nathante.sif spark-submit --master spark://$(hostname):7077 tfidf.py authors --topN=100000 --inpath=/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet --outpath=/gscratch/scrubbed/comdata/reddit_similarity/tfidf/comment_authors_100k.parquet
singularity exec /gscratch/comdata/users/nathante/containers/nathante.sif stop-all.sh
