from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from similarities_helper import build_weekly_tfidf_dataset
import pandas as pd


## TODO:need to exclude automoderator / bot posts.
## TODO:need to exclude better handle hyperlinks. 

spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet")

include_subs = pd.read_csv("/gscratch/comdata/output/reddit_similarity/subreddits_by_num_comments.csv")

include_subs = set(include_subs.loc[include_subs.comments_rank <= 25000]['subreddit'])

# remove [deleted] and AutoModerator (TODO remove other bots)
# df = df.filter(df.author != '[deleted]')
# df = df.filter(df.author != 'AutoModerator')

df = build_weekly_tfidf_dataset(df, include_subs, 'term')


df.write.parquet('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet', mode='overwrite', compression='snappy')
spark.stop()

