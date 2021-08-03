all: /gscratch/comdata/output/reddit_density/comment_terms_10000.feather /gscratch/comdata/output/reddit_density/comment_authors_10000.feather /gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10000.feather

/gscratch/comdata/output/reddit_density/comment_terms_10000.feather:overlap_density.py /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather /gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather
	start_spark_and_run.sh 1 overlap_density.py terms --inpath="/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather" --outpath="/gscratch/comdata/output/reddit_density/comment_terms_10000.feather" --agg=pd.DataFrame.sum

/gscratch/comdata/output/reddit_density/comment_authors_10000.feather:overlap_density.py /gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather
	start_spark_and_run.sh 1 overlap_density.py authors --inpath="/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather" --outpath="/gscratch/comdata/output/reddit_density/comment_authors_10000.feather" --agg=pd.DataFrame.sum

/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10000.feather: overlap_density.py /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet
	start_spark_and_run.sh 1 overlap_density.py authors --inpath="/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet" --outpath="/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10000.feather" --agg=pd.DataFrame.sum

/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10K_LSI/850.feather: overlap_density.py /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/850.feather
	start_spark_and_run.sh 1 overlap_density.py authors --inpath="/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/850.feather" --outpath="/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10K_LSI/850.feather" --agg=pd.DataFrame.sum

/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather: overlap_density.py /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather
	start_spark_and_run.sh 1 overlap_density.py authors --inpath="/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/600.feather" --outpath="/gscratch/comdata/output/reddit_density/subreddit_author_tf_similarities_10K_LSI/600.feather" --agg=pd.DataFrame.sum
