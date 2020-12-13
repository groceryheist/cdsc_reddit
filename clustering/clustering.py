#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
import fire

def affinity_clustering(similarities, output, damping=0.9, max_iter=100000, convergence_iter=30, preference_quantile=0.5, random_state=1968, verbose=True):
    '''
    similarities: feather file with a dataframe of similarity scores
    preference_quantile: parameter controlling how many clusters to make. higher values = more clusters. 0.85 is a good value with 3000 subreddits.
    damping: parameter controlling how iterations are merged. Higher values make convergence faster and more dependable. 0.85 is a good value for the 10000 subreddits by author. 
    '''

    df = pd.read_feather(similarities)
    n = df.shape[0]
    mat = np.array(df.drop('subreddit',1))
    mat[range(n),range(n)] = 1

    preference = np.quantile(mat,preference_quantile)

    print(f"preference is {preference}")

    print("data loaded")

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

    cluster_data = pd.DataFrame({'subreddit': df.subreddit,'cluster':clustering.labels_})

    cluster_sizes = cluster_data.groupby("cluster").count()
    print(f"the largest cluster has {cluster_sizes.subreddit.max()} members")

    print(f"the median cluster has {cluster_sizes.subreddit.median()} members")

    print(f"{(cluster_sizes.subreddit==1).sum()} clusters have 1 member")

    cluster_data.to_feather(output)

if __name__ == "__main__":
    fire.Fire(affinity_clustering)
