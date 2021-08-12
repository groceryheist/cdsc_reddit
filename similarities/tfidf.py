import fire
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from similarities_helper import tfidf_dataset, build_weekly_tfidf_dataset, select_topN_subreddits

def _tfidf_wrapper(func, inpath, outpath, topN, term_colname, exclude, included_subreddits):
    spark = SparkSession.builder.getOrCreate()y

    df = spark.read.parquet(inpath)

    df = df.filter(~ f.col(term_colname).isin(exclude))

    if included_subreddits is not None:
        include_subs = set(map(str.strip,open(included_subreddits)))
    else:
        include_subs = select_topN_subreddits(topN)

    dfwriter = func(df, include_subs, term_colname)

    dfwriter.parquet(outpath,mode='overwrite',compression='snappy')
    spark.stop()

def tfidf(inpath, outpath, topN, term_colname, exclude, included_subreddits):
    return _tfidf_wrapper(tfidf_dataset, inpath, outpath, topN, term_colname, exclude, included_subreddits)

def tfidf_weekly(inpath, outpath, topN, term_colname, exclude, included_subreddits):
    return _tfidf_wrapper(build_weekly_tfidf_dataset, inpath, outpath, topN, term_colname, exclude, included_subreddits)

def tfidf_authors(inpath="/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                  outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet',
                  topN=None,
                  included_subreddits=None):

    return tfidf(inpath,
                 outpath,
                 topN,
                 'author',
                 ['[deleted]','AutoModerator'],
                 included_subreddits=included_subreddits
                 )

def tfidf_terms(inpath="/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                outpath='/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet',
                topN=None,
                included_subreddits=None):

    return tfidf(inpath,
                 outpath,
                 topN,
                 'term',
                 [],
                 included_subreddits=included_subreddits
                 )

def tfidf_authors_weekly(inpath="/gscratch/comdata/output/reddit_ngrams/comment_authors.parquet",
                         outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_authors.parquet',
                         topN=None,
                         included_subreddits=None):

    return tfidf_weekly(inpath,
                        outpath,
                        topN,
                        'author',
                        ['[deleted]','AutoModerator'],
                        included_subreddits=included_subreddits
                        )

def tfidf_terms_weekly(inpath="/gscratch/comdata/output/reddit_ngrams/comment_terms.parquet",
                       outpath='/gscratch/comdata/output/reddit_similarity/tfidf_weekly/comment_terms.parquet',
                       topN=None,
                       included_subreddits=None):


    return tfidf_weekly(inpath,
                        outpath,
                        topN,
                        'term',
                        [],
                        included_subreddits=included_subreddits
                        )


if __name__ == "__main__":
    fire.Fire({'authors':tfidf_authors,
               'terms':tfidf_terms,
               'authors_weekly':tfidf_authors_weekly,
               'terms_weekly':tfidf_terms_weekly})
