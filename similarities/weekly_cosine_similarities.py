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
from similarities_helper import *
from multiprocessing import Pool, cpu_count
from functools import partial


def _week_similarities(week, simfunc, tfidf_path, term_colname, min_df, max_df, included_subreddits, topN, outdir:Path):
    term = term_colname
    term_id = term + '_id'
    term_id_new = term + '_id_new'
    print(f"loading matrix: {week}")
    entries, subreddit_names = reindex_tfidf(infile = tfidf_path,
                                             term_colname=term_colname,
                                             min_df=min_df,
                                             max_df=max_df,
                                             included_subreddits=included_subreddits,
                                             topN=topN,
                                             week=week)
    mat = csr_matrix((entries[tfidf_colname],(entries[term_id_new], entries.subreddit_id_new)))
    print('computing similarities')
    sims = column_similarities(mat)
    del mat
    sims = pd.DataFrame(sims.todense())
    sims = sims.rename({i: sr for i, sr in enumerate(subreddit_names.subreddit.values)}, axis=1)
    sims['_subreddit'] = names.subreddit.values
    outfile = str(Path(outdir) / str(week))
    write_weekly_similarities(outfile, sims, week, names)

def pull_weeks(batch):
    return set(batch.to_pandas()['week'])

#tfidf = spark.read.parquet('/gscratch/comdata/users/nathante/subreddit_tfidf_weekly.parquet')
def cosine_similarities_weekly(tfidf_path, outfile, term_colname, min_df = None, max_df=None, included_subreddits = None, topN = 500):
    print(outfile)
    tfidf_ds = ds.dataset(tfidf_path)
    tfidf_ds = tfidf_ds.to_table(columns=["week"])
    batches = tfidf_ds.to_batches()

    with Pool(cpu_count()) as pool:
        weeks = set(chain( * pool.imap_unordered(pull_weeks,batches)))

    weeks = sorted(weeks)
    # do this step in parallel if we have the memory for it.
    # should be doable with pool.map

    print(f"computing weekly similarities")
    week_similarities_helper = partial(_week_similarities,simfunc=column_similarities, tfidf_path=tfidf_path, term_colname=term_colname, outdir=outfile, min_df=min_df,max_df=max_df,included_subreddits=included_subreddits,topN=topN)

    with Pool(cpu_count()) as pool: # maybe it can be done with 40 cores on the huge machine?
        list(pool.map(week_similarities_helper,weeks))

def author_cosine_similarities_weekly(outfile, min_df=2, max_df=None, included_subreddits=None, topN=500):
    return cosine_similarities_weekly('/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet',
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
