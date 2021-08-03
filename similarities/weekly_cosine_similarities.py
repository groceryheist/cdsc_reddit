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
from similarities_helper import pull_tfidf, column_similarities, write_weekly_similarities
from scipy.sparse import csr_matrix
from multiprocessing import Pool, cpu_count
from functools import partial

# infile = "/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_test.parquet"
# tfidf_path = infile 
# min_df=None
# max_df = None
# topN=100
# term_colname='author'
# outfile = '/gscratch/comdata/output/reddit_similarity/weekly/comment_authors_test.parquet'
# included_subreddits=None

def _week_similarities(week, simfunc, tfidf_path, term_colname, min_df, max_df, included_subreddits, topN, outdir:Path, subreddit_names, nterms):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'
    print(f"loading matrix: {week}")

    entries = pull_tfidf(infile = tfidf_path,
                         term_colname=term_colname,
                         min_df=min_df,
                         max_df=max_df,
                         included_subreddits=included_subreddits,
                         topN=topN,
                         week=week.isoformat(),
                         rescale_idf=False)
    
    tfidf_colname='tf_idf'
    # if the max subreddit id we found is less than the number of subreddit names then we have to fill in 0s
    mat = csr_matrix((entries[tfidf_colname],(entries[term_id_new]-1, entries.subreddit_id_new-1)),shape=(nterms,subreddit_names.shape[0]))

    print('computing similarities')
    sims = simfunc(mat.T)
    del mat
    sims = pd.DataFrame(sims)
    sims = sims.rename({i: sr for i, sr in enumerate(subreddit_names.subreddit.values)}, axis=1)
    sims['_subreddit'] = subreddit_names.subreddit.values
    outfile = str(Path(outdir) / str(week))
    write_weekly_similarities(outfile, sims, week, subreddit_names)

def pull_weeks(batch):
    return set(batch.to_pandas()['week'])

#tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_weekly.parquet')
def cosine_similarities_weekly(tfidf_path, outfile, term_colname, min_df = None, max_df=None, included_subreddits = None, topN = 500):
    print(outfile)
    # do this step in parallel if we have the memory for it.
    # should be doable with pool.map

    spark = SparkSession.builder.getOrCreate()
    df = spark.read.parquet(tfidf_path)
    subreddit_names = df.select(['subreddit','subreddit_id']).distinct().toPandas()
    subreddit_names = subreddit_names.sort_values("subreddit_id")
    nterms = df.select(f.max(f.col(term_colname + "_id")).alias('max')).collect()[0].max
    weeks = df.select(f.col("week")).distinct().toPandas().week.values
    spark.stop()

    print(f"computing weekly similarities")
    week_similarities_helper = partial(_week_similarities,simfunc=column_similarities, tfidf_path=tfidf_path, term_colname=term_colname, outdir=outfile, min_df=min_df,max_df=max_df,included_subreddits=included_subreddits,topN=topN, subreddit_names=subreddit_names,nterms=nterms)

    pool = Pool(cpu_count())
    
    list(pool.imap(week_similarities_helper,weeks))
    pool.close()
    #    with Pool(cpu_count()) as pool: # maybe it can be done with 40 cores on the huge machine?


def author_cosine_similarities_weekly(outfile, min_df=2, max_df=None, included_subreddits=None, topN=500):
    return cosine_similarities_weekly('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors_test.parquet',
                                      outfile,
                                      'author',
                                      min_df,
                                      max_df,
                                      included_subreddits,
                                      topN)

def term_cosine_similarities_weekly(outfile, min_df=None, max_df=None, included_subreddits=None, topN=500):
        return cosine_similarities_weekly('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet',
                                          outfile,
                                          'term',
                                          min_df,
                                          max_df,
                                          included_subreddits,
                                          topN)

if __name__ == "__main__":
    fire.Fire({'authors':author_cosine_similarities_weekly,
               'terms':term_cosine_similarities_weekly})
