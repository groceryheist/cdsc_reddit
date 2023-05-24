all: ../../data/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather

../../data/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather: overlap_density.py ../../data/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather
	../start_spark_and_run.sh 1 overlap_density.py authors --inpath="../../data/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather" --outpath="../../data/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather" --agg=pd.DataFrame.sum

../../data/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather:
	$(MAKE) -C ../similarity
