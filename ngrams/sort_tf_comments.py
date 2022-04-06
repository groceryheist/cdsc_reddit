#!/usr/bin/env python3

from pyspark.sql import functions as f
from pyspark.sql import SparkSession
import fire

def main(inparquet, outparquet, colname):
    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(inparquet)

    df = df.repartition(2000,colname)
    df = df.sort([colname,'week','subreddit'])
    df = df.sortWithinPartitions([colname,'week','subreddit'])

    df.write.parquet(outparquet,mode='overwrite',compression='snappy')

if __name__ == '__main__':
    fire.Fire(main)
