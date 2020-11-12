from pyspark.sql import SparkSession
from similarities_helper import build_tfidf_dataset

spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp")

include_subs = set(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"))
include_subs = {s.strip('\n') for s in include_subs}

# remove [deleted] and AutoModerator (TODO remove other bots)
df = df.filter(df.author != '[deleted]')
df = df.filter(df.author != 'AutoModerator')

df = build_tfidf_dataset(df, include_subs, 'author')

df.write.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_authors.parquet',mode='overwrite',compression='snappy')

spark.stop()
