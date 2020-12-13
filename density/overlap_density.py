import pandas as pd
from pandas.core.groupby import DataFrameGroupBy as GroupBy
import fire
import numpy as np

def overlap_density(inpath, outpath, agg = pd.DataFrame.sum):
    df = pd.read_feather(inpath)
    df = df.drop('subreddit',1)
    np.fill_diagonal(df.values,0)
    df = agg(df, 0).reset_index()
    df = df.rename({0:'overlap_density'},axis='columns')
    df.to_feather(outpath)
    return df

def overlap_density_weekly(inpath, outpath, agg = GroupBy.sum):
    df = pd.read_parquet(inpath)
    # exclude the diagonal
    df = df.loc[df.subreddit != df.variable]
    res = agg(df.groupby(['subreddit','week'])).reset_index()
    res.to_feather(outpath)
    return res

def author_overlap_density(inpath="/gscratch/comdata/output/reddit_similarity/comment_authors_10000.feather",
                           outpath="/gscratch/comdata/output/reddit_density/comment_authors_10000.feather", agg=pd.DataFrame.sum):
    if type(agg) == str:
        agg = eval(agg)

    overlap_density(inpath, outpath, agg)

def term_overlap_density(inpath="/gscratch/comdata/output/reddit_similarity/comment_terms_10000.feather",
                         outpath="/gscratch/comdata/output/reddit_density/comment_term_similarity_10000.feather", agg=pd.DataFrame.sum):

    if type(agg) == str:
        agg = eval(agg)

    overlap_density(inpath, outpath, agg)

def author_overlap_density_weekly(inpath="/gscratch/comdata/output/reddit_similarity/subreddit_authors_10000_weekly.parquet",
                                  outpath="/gscratch/comdata/output/reddit_density/comment_authors_10000_weekly.feather", agg=GroupBy.sum):
    if type(agg) == str:
        agg = eval(agg)

    overlap_density_weekly(inpath, outpath, agg)

def term_overlap_density_weekly(inpath="/gscratch/comdata/output/reddit_similarity/comment_terms_10000_weekly.parquet",
                                outpath="/gscratch/comdata/output/reddit_density/comment_terms_10000_weekly.parquet", agg=GroupBy.sum):
    if type(agg) == str:
        agg = eval(agg)

    overlap_density_weekly(inpath, outpath, agg)


if __name__ == "__main__":
    fire.Fire({'authors':author_overlap_density,
               'terms':term_overlap_density,
               'author_weekly':author_overlap_density_weekly,
               'term_weekly':term_overlap_density_weekly})
