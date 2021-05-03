from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

def sim_to_dist(mat):
    dist = 1-mat
    dist[dist < 0] = 0
    np.fill_diagonal(dist,0)
    return dist

def process_clustering_result(clustering, subreddits):

    if hasattr(clustering,'n_iter_'):
        print(f"clustering took {clustering.n_iter_} iterations")

    clusters = clustering.labels_

    print(f"found {len(set(clusters))} clusters")

    cluster_data = pd.DataFrame({'subreddit': subreddits,'cluster':clustering.labels_})

    cluster_sizes = cluster_data.groupby("cluster").count().reset_index()
    print(f"the largest cluster has {cluster_sizes.loc[cluster_sizes.cluster!=-1].subreddit.max()} members")

    print(f"the median cluster has {cluster_sizes.subreddit.median()} members")

    print(f"{(cluster_sizes.subreddit==1).sum()} clusters have 1 member")

    print(f"{(cluster_sizes.loc[cluster_sizes.cluster==-1,['subreddit']])} subreddits are in cluster -1",flush=True)

    return cluster_data


@dataclass
class clustering_result:
    outpath:Path
    max_iter:int
    silhouette_score:float
    alt_silhouette_score:float
    name:str
    n_clusters:int

def read_similarity_mat(similarities, use_threads=True):
    df = pd.read_feather(similarities, use_threads=use_threads)
    mat = np.array(df.drop('_subreddit',1))
    n = mat.shape[0]
    mat[range(n),range(n)] = 1
    return (df._subreddit,mat)
