import pandas as pd
import numpy as np
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from .choose_clusters import load_clusters, load_densities
import fire
from pathlib import Path

def build_cluster_timeseries(term_clusters_path="/gscratch/comdata/output/reddit_clustering/comment_terms_10000.feather",
         author_clusters_path="/gscratch/comdata/output/reddit_clustering/comment_authors_10000.feather",
         term_densities_path="/gscratch/comdata/output/reddit_density/comment_terms_10000.feather",
         author_densities_path="/gscratch/comdata/output/reddit_density/comment_authors_10000.feather",
         output="data/subreddit_timeseries.parquet"):


    clusters = load_clusters(term_clusters_path, author_clusters_path)
    densities = load_densities(term_densities_path, author_densities_path)
    
    spark = SparkSession.builder.getOrCreate()
    
    df = spark.read.parquet("/gscratch/comdata/output/reddit_comments_by_subreddit.parquet")
    
    df = df.withColumn('week', f.date_trunc('week', f.col("CreatedAt")))
    
    # time of unique authors by series by week
    ts = df.select(['subreddit','week','author']).distinct().groupby(['subreddit','week']).count()
    
    ts = ts.repartition('subreddit')
    spk_clusters = spark.createDataFrame(clusters)
    
    ts = ts.join(spk_clusters, on='subreddit', how='inner')
    spk_densities = spark.createDataFrame(densities)
    ts = ts.join(spk_densities, on='subreddit', how='inner')
    ts.write.parquet(output, mode='overwrite')

if __name__ == "__main__":
    fire.Fire(build_cluster_timeseries)
