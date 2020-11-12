from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from similarities_helper import build_tfidf_dataset

## TODO:need to exclude automoderator / bot posts.
## TODO:need to exclude better handle hyperlinks. 

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/users/nathante/reddit_tfidf_test.parquet_temp")

include_subs = set(open("/gscratch/comdata/users/nathante/cdsc-reddit/top_25000_subs_by_comments.txt"))
include_subs = {s.strip('\n') for s in include_subs}

df = build_tfidf_dataset(df, include_subs, 'term')

df.write.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf.parquet',mode='overwrite',compression='snappy')
spark.stop()
