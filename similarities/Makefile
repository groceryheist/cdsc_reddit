all: /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/comment_terms.parquet

# all: /gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.parquet /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_10000.parquet /gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet


# /gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
# 	start_spark_and_run.sh 1 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors_25000.feather

/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet: tfidf.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_terms.parquet /gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv
	start_spark_and_run.sh 1 tfidf.py terms --topN=10000

/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet: tfidf.py similarities_helper.py /gscratch/comdata/output/reddit_ngrams/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv
	start_spark_and_run.sh 1 tfidf.py authors --topN=10000

/gscratch/comdata/output/reddit_similarity/comment_authors_10000.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet 
	start_spark_and_run.sh 1 cosine_similarities.py author --outfile=/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather

/gscratch/comdata/output/reddit_similarity/comment_terms.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet
	start_spark_and_run.sh 1 cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather

# /gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet: cosine_similarities.py /gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet
# 	start_spark_and_run.sh 1 weekly_cosine_similarities.py term --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_comment_terms_10000_weely.parquet

/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet: cosine_similarities.py similarities_helper.py /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet /gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet
	start_spark_and_run.sh 1 cosine_similarities.py author-tf --outfile=/gscratch/comdata/output/reddit_similarity/subreddit_author_tf_similarities_10000.parquet
