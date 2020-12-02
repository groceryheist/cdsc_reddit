from pyspark.sql import SparkSession
from similarities_helper import build_tfidf_dataset
import pandas as pd

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp")

include_subs = pd.read_csv("/gscratch/comdata/users/nathante/cdsc-reddit/subreddits_by_num_comments.csv")

#include_subs = set(include_subs.loc[include_subs.comments_rank < 300]['subreddit'])

# remove [deleted] and AutoModerator (TODO remove other bots)
df = df.filter(df.author != '[deleted]')
df = df.filter(df.author != 'AutoModerator')

df = build_tfidf_dataset(df, include_subs, 'author')

df.write.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_authors.parquet',mode='overwrite',compression='snappy')

spark.stop()
