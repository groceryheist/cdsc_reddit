from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from datetime import datetime
from pathlib import Path

spark = SparkSession.builder.getOrCreate()
conf = spark.sparkContext.getConf()

submissions = spark.read.parquet("../../data/reddit_submissions_by_subreddit.parquet")

submissions = submissions.filter(f.col("CreatedAt") <= datetime(2020,4,13))

prop_nsfw = submissions.select(['subreddit','over_18']).groupby('subreddit').agg(f.mean(f.col('over_18').astype('double')).alias('prop_nsfw'))

df = spark.read.parquet("../../data/reddit_comments_by_subreddit.parquet")
df = df.filter(f.col("CreatedAt") <= datetime(2020,4,13))
# remove /u/ pages
df = df.filter(~df.subreddit.like("u_%"))

df = df.groupBy('subreddit').agg(f.count('id').alias("n_comments"))

df = df.join(prop_nsfw,on='subreddit')
df = df.filter(df.prop_nsfw < 0.5)

win = Window.orderBy(f.col('n_comments').desc())
df = df.withColumn('comments_rank', f.rank().over(win))

df = df.toPandas()

df = df.sort_values("n_comments")

outpath = Path("../../data/reddit_similarity/subreddits_by_num_comments_nonsfw.csv")
outpath.parent.mkdir(exist_ok=True, parents=True)
df.to_csv(str(outpath), index=False)
