from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
import numpy as np
import pyarrow
import pandas as pd
import fire
from itertools import islice
from pathlib import Path
from similarities_helper import *
from multiprocessing import pool

def _week_similarities(tempdir, term_colname, week):
        print(f"loading matrix: {week}")
        mat = read_tfidf_matrix_weekly(tempdir.name, term_colname, week)
        print('computing similarities')
        sims = column_similarities(mat)
        del mat

        names = subreddit_names.loc[subreddit_names.week == week]
        sims = pd.DataFrame(sims.todense())

        sims = sims.rename({i: sr for i, sr in enumerate(names.subreddit.values)}, axis=1)
        sims['_subreddit'] = names.subreddit.values

        write_weekly_similarities(outfile, sims, week, names)

#tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_weekly.parquet')
def cosine_similarities_weekly(tfidf_path, outfile, term_colname, min_df = None, included_subreddits = None, topN = 500):
    spark = SparkSession.builder.getOrCreate()
    conf = spark.sparkContext.getConf()
    print(outfile)
    tfidf = spark.read.parquet(tfidf_path)
    
    if included_subreddits is None:
        included_subreddits = select_topN_subreddits(topN)
    else:
        included_subreddits = set(open(included_subreddits))

    print(f"computing weekly similarities for {len(included_subreddits)} subreddits")

    print("creating temporary parquet with matrix indicies")
    tempdir = prep_tfidf_entries_weekly(tfidf, term_colname, min_df, included_subreddits)

    tfidf = spark.read.parquet(tempdir.name)

    # the ids can change each week.
    subreddit_names = tfidf.select(['subreddit','subreddit_id_new','week']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id_new")
    subreddit_names['subreddit_id_new'] = subreddit_names['subreddit_id_new'] - 1
    spark.stop()

    weeks = sorted(list(subreddit_names.week.drop_duplicates()))
    # do this step in parallel if we have the memory for it.
    # should be doable with pool.map

    def week_similarities_helper(week):
        _week_similarities(tempdir, term_colname, week)

    with Pool(40) as pool: # maybe it can be done with 40 cores on the huge machine?
        list(pool.map(weeks,week_similarities_helper))

def author_cosine_similarities_weekly(outfile, min_df=2 , included_subreddits=None, topN=500):
    return cosine_similarities_weekly('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_100k.parquet',
                                      outfile,
                                      'author',
                                      min_df,
                                      included_subreddits,
                                      topN)

def term_cosine_similarities_weekly(outfile, min_df=None, included_subreddits=None, topN=500):
    return cosine_similarities_weekly('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms_100k.parquet',
                                      outfile,
                                      'term',
                                      min_df,
                                      included_subreddits,
                                      topN)

if __name__ == "__main__":
    fire.Fire({'authors':author_cosine_similarities_weekly,
               'terms':term_cosine_similarities_weekly})
