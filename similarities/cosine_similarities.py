import pandas as pd
import fire
from pathlib import Path
from similarities_helper import similarities, column_similarities

def cosine_similarities(infile, term_colname, outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, exclude_phrases=False, from_date=None, to_date=None, tfidf_colname='tf_idf'):

    return similarities(infile=infile, simfunc=column_similarities, term_colname=term_colname, outfile=outfile, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, exclude_phrases=exclude_phrases,from_date=from_date, to_date=to_date, tfidf_colname=tfidf_colname)


def term_cosine_similarities(outfile, min_df=None, max_df=None, included_subreddits=None, topN=500, exclude_phrases=False, from_date=None, to_date=None):
    return cosine_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_terms.parquet',
                               'term',
                               outfile,
                               min_df,
                               max_df,
                               included_subreddits,
                               topN,
                               exclude_phrases,
                               from_date,
                               to_date)

def author_cosine_similarities(outfile, min_df=2, max_df=None, included_subreddits=None, topN=10000, from_date=None, to_date=None):
    return cosine_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet',
                               'author',
                               outfile,
                               min_df,
                               max_df,
                               included_subreddits,
                               topN,
                               exclude_phrases=False,
                               from_date=from_date,
                               to_date=to_date)

def author_tf_similarities(outfile, min_df=2, max_df=None, included_subreddits=None, topN=10000, from_date=None, to_date=None):
    return cosine_similarities('/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet',
                               'author',
                               outfile,
                               min_df,
                               max_df,
                               included_subreddits,
                               topN,
                               exclude_phrases=False,
                               from_date=from_date,
                               to_date=to_date,
                               tfidf_colname='relative_tf')


if __name__ == "__main__":
    fire.Fire({'term':term_cosine_similarities,
               'author':author_cosine_similarities,
               'author-tf':author_tf_similarities})

