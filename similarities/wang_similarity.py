from similarities_helper import similarities
import numpy as np
import fire 

def wang_similarity(mat):
    non_zeros = (mat != 0).astype(np.float32)
    intersection = non_zeros.T @ non_zeros
    return intersection


infile="/gscratch/comdata/output/reddit_similarity/tfidf/comment_authors.parquet"; outfile="/gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather"; min_df=1; included_subreddits=None; topN=10000; exclude_phrases=False; from_date=None; to_date=None
    
def wang_overlaps(infile, outfile="/gscratch/comdata/output/reddit_similarity/wang_similarity_10000.feather", min_df=1, max_df=None, included_subreddits=None, topN=10000, exclude_phrases=False, from_date=None, to_date=None):

    return similarities(infile=infile, simfunc=wang_similarity, term_colname='author', outfile=outfile, min_df=min_df, max_df=max_df, included_subreddits=included_subreddits, topN=topN, exclude_phrases=exclude_phrases, from_date=from_date, to_date=to_date)

if __name__ == "__main__":
    fire.Fire(wang_overlaps)
