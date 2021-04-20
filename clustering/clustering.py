#!/usr/bin/env python3
# TODO: replace prints with logging.
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
import fire
from pathlib import Path

def read_similarity_mat(similarities, use_threads=True):
    df = pd.read_feather(similarities, use_threads=use_threads)
    mat = np.array(df.drop('_subreddit',1))
    n = mat.shape[0]
    mat[range(n),range(n)] = 1
    return (df._subreddit,mat)

def affinity_clustering(similarities, *args, **kwargs):
    subreddits, mat = read_similarity_mat(similarities)
    return _affinity_clustering(mat, subreddits, *args, **kwargs)

def _affinity_clustering(mat, subreddits, output, damping=0.9, max_iter=100000, convergence_iter=30, preference_quantile=0.5, random_state=1968, verbose=True):
    '''
    similarities: feather file with a dataframe of similarity scores
    preference_quantile: parameter controlling how many clusters to make. higher values = more clusters. 0.85 is a good value with 3000 subreddits.
    damping: parameter controlling how iterations are merged. Higher values make convergence faster and more dependable. 0.85 is a good value for the 10000 subreddits by author. 
    '''
    print(f"damping:{damping}; convergenceIter:{convergence_iter}; preferenceQuantile:{preference_quantilne}")

    preference = np.quantile(mat,preference_quantile)

    print(f"preference is {preference}")
    print("data loaded")
    sys.stdout.flush()
    clustering = AffinityPropagation(damping=damping,
                                     max_iter=max_iter,
                                     convergence_iter=convergence_iter,
                                     copy=False,
                                     preference=preference,
                                     affinity='precomputed',
                                     verbose=verbose,
                                     random_state=random_state).fit(mat)


    print(f"clustering took {clustering.n_iter_} iterations")
    clusters = clustering.labels_

    print(f"found {len(set(clusters))} clusters")

    cluster_data = pd.DataFrame({'subreddit': subreddits,'cluster':clustering.labels_})

    cluster_sizes = cluster_data.groupby("cluster").count()
    print(f"the largest cluster has {cluster_sizes.subreddit.max()} members")

    print(f"the median cluster has {cluster_sizes.subreddit.median()} members")

    print(f"{(cluster_sizes.subreddit==1).sum()} clusters have 1 member")

    sys.stdout.flush()
    cluster_data.to_feather(output)
    print(f"saved {output}")
    return clustering

if __name__ == "__main__":
    fire.Fire(affinity_clustering)
