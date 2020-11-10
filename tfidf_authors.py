from pyspark.sql import SparkSession
from similarities_helper import build_tfidf_dataset

## TODO:need to exclude automoderator / bot posts.
## TODO:need to exclude better handle hyperlinks. 
spark = SparkSession.builder.getOrCreate()

df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test_authors.parquet_temp/part-00000-d61007de-9cbe-4970-857f-b9fd4b35b741-c000.snappy.parquet")

include_subs = set(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"))
include_subs = {s.strip('\n') for s in include_subs}
df = df.filter(df.author != '[deleted]')
df = df.filter(df.author != 'AutoModerator')

df = build_tfidf_dataset(df, include_subs, 'author')

df.cache()

df.write.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_authors.parquet',mode='overwrite',compression='snappy')
