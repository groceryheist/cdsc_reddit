from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark.mllib.linalg.distributed import RowMatrix, CoordinateMatrix
import numpy as np
import pyarrow
import pandas as pd
import fire
from itertools import islice
from pathlib import Path
from similarities_helper import cosine_similarities

spark = SparkSession.builder.getOrCreate()
conf = spark.sparkContext.getConf()

df = spark.read.parquet("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet")

# remove /u/ pages
df = df.filter(~df.subreddit.like("u_%"))

df = df.groupBy('subreddit').agg(f.count('id').alias("n_comments"))

win = Window.orderBy(f.col('n_comments').desc())
df = df.withColumn('comments_rank',f.rank().over(win))

df = df.toPandas()

df = df.sort_values("n_comments")

df.to_csv('/gscratch/comdata/users/nathante/cdsc-reddit/subreddits_by_num_comments.csv',index=False)
