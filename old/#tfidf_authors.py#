from pyspark.sql import SparkSession
from similarities_helper import build_tfidf_dataset
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet")

include_subs = pd.read_csv("/gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv")

include_subs = set(include_subs.loc[include_subs.comments_rank <= 25000]['subreddit'])

# remove [deleted] and AutoModerator (TODO remove other bots)
df = df.filter(df.author != '[deleted]')
df = df.filter(df.author != 'AutoModerator')

df = build_tfidf_dataset(df, include_subs, 'author')

df.write.parquet('/gscratch/comdata/output/reddit_similarity/tfidf/subreddit_comment_authors.parquet',mode='overwrite',compression='snappy')

spark.stop()
