#!/usr/bin/env python3
from pyspark.sql import functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
import numpy as np
import pyarrow
import pyarrow.dataset as ds
import pandas as pd
import fire
from itertools import islice, chain
from pathlib import Path
from similarities_helper import pull_tfidf, column_similarities, write_weekly_similarities, lsi_column_similarities
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from functools import partial
import pickle

# tfidf_path = "/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/similarity_weekly/comment_authors_tfidf.parquet"
# #tfidf_path = "/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data//comment_authors_compex.parquet"
# min_df=2
# included_subreddits="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/included_subreddits.txt"
# max_df = None
# topN=100
# term_colname='author'
# # outfile = '/gscratch/comdata/output/reddit_similarity/weekly/comment_authors_test.parquet'
# # included_subreddits=None
outfile="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/similarity_weekly/comment_authors.parquet"; infile="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/tfidf_weekly/comment_authors_tfidf.parquet"; included_subreddits="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/included_subreddits.txt"; lsi_model="/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/similarity/comment_authors_compex_LSI/2000_authors_LSIMOD.pkl"; n_components=1500; algorithm="randomized"; term_colname='author'; tfidf_path=infile; random_state=1968;

# static_tfidf = "/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/tfidf/comment_authors_compex.parquet"
# dftest = spark.read.parquet(static_tfidf)

def _week_similarities(week, simfunc, tfidf_path, term_colname, included_subreddits, outdir:Path, subreddit_names, nterms, topN=None, min_df=None, max_df=None):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'
    print(f"loading matrix: {week}")

    entries = pull_tfidf(infile = tfidf_path,
                         term_colname=term_colname,
                         included_subreddits=included_subreddits,
                         topN=topN,
                         week=week.isoformat(),
                         rescale_idf=False)
    
    tfidf_colname='tf_idf'
    # if the max subreddit id we found is less than the number of subreddit names then we have to fill in 0s
    mat = csr_matrix((entries[tfidf_colname],(entries[term_id_new]-1, entries.subreddit_id_new-1)),shape=(nterms,subreddit_names.shape[0]))
    print('computing similarities')
    print(simfunc)
    sims = simfunc(mat)
    del mat
    sims = next(sims)[0]
    sims = pd.DataFrame(sims)
    sims = sims.rename({i: sr for i, sr in enumerate(subreddit_names.subreddit.values)}, axis=1)
    sims['_subreddit'] = subreddit_names.subreddit.values
    outfile = str(Path(outdir) / str(week))
    write_weekly_similarities(outfile, sims, week, subreddit_names)

def pull_weeks(batch):
    return set(batch.to_pandas()['week'])

# This requires a prefit LSI model, since we shouldn't fit different LSI models for every week. 
def cosine_similarities_weekly_lsi(*args, n_components=100, lsi_model=None, **kwargs):
    print(args)
    print(kwargs)
    term_colname= kwargs.get('term_colname')
    # lsi_model = "/gscratch/comdata/users/nathante/competitive_exclusion_reddit/data/similarity/comment_authors_compex_LSI/1000_author_LSIMOD.pkl"

    lsi_model = pickle.load(open(lsi_model,'rb'))
    #simfunc = partial(lsi_column_similarities,n_components=n_components,random_state=random_state,algorithm='randomized',lsi_model=lsi_model)
    simfunc = partial(lsi_column_similarities,n_components=n_components,random_state=kwargs.get('random_state'),lsi_model=lsi_model)

    return cosine_similarities_weekly(*args, simfunc=simfunc, **kwargs)

#tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_weekly.parquet')
def cosine_similarities_weekly(tfidf_path, outfile, term_colname, included_subreddits = None, topN = None, simfunc=column_similarities, min_df=None,max_df=None):
    print(outfile)
    # do this step in parallel if we have the memory for it.
    # should be doable with pool.map

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(tfidf_path)

    # load subreddits + topN
        
    subreddit_names = df.select(['subreddit','subreddit_id']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id")
    nterms = df.select(f.max(f.col(term_colname + "_id")).alias('max')).collect()[0].max
    weeks = df.select(f.col("week")).distinct().toPandas().week.values
    spark.stop()

    print(f"computing weekly similarities")
    week_similarities_helper = partial(_week_similarities,simfunc=simfunc, tfidf_path=tfidf_path, term_colname=term_colname, outdir=outfile, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=None, subreddit_names=subreddit_names,nterms=nterms)

    for week in weeks:
        week_similarities_helper(week)
    # pool = Pool(cpu_count())
        
    # list(pool.imap(week_similarities_helper, weeks))
    # pool.close()
    #    with Pool(cpu_count()) as pool: # maybe it can be done with 40 cores on the huge machine?


def author_cosine_similarities_weekly(outfile, infile='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_test.parquet', min_df=2, max_df=None, included_subreddits=None, topN=500):
    return cosine_similarities_weekly(infile,
                                      outfile,
                                      'author',
                                      max_df,
                                      included_subreddits,
                                      topN,
                                      min_df=2
)

def term_cosine_similarities_weekly(outfile, infile='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet', min_df=None, max_df=None, included_subreddits=None, topN=None):
        return cosine_similarities_weekly(infile,
                                          outfile,
                                          'term',
                                          min_df,
                                          max_df,
                                          included_subreddits,
                                          topN)


def author_cosine_similarities_weekly_lsi(outfile, infile = '/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_test.parquet', included_subreddits=None, n_components=100,lsi_model=None):
    return cosine_similarities_weekly_lsi(infile,
                                          outfile,
                                          'author',
                                          included_subreddits=included_subreddits,
                                          n_components=n_components,
                                          lsi_model=lsi_model
                                          )


def term_cosine_similarities_weekly_lsi(outfile, infile = '/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet', included_subreddits=None, n_components=100,lsi_model=None):
        return cosine_similarities_weekly_lsi(infile,
                                              outfile,
                                              'term',
                                              included_subreddits=included_subreddits,
                                              n_components=n_components,
                                              lsi_model=lsi_model,
                                              )

if __name__ == "__main__":
    fire.Fire({'authors':author_cosine_similarities_weekly,
               'terms':term_cosine_similarities_weekly,
               'authors-lsi':author_cosine_similarities_weekly_lsi,
               'terms-lsi':term_cosine_similarities_weekly_lsi
               })

