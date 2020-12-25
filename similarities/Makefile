all: /gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet

/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_25000.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet
	start_spark_and_run.sh 1 cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_25000.feather

/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
	start_spark_and_run.sh 1 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.feather

/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
	start_spark_and_run.sh 1 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.feather

/gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
	start_spark_and_run.sh 1 weekly_cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_10000_weely.parquet
