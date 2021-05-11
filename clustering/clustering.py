#!/usr/bin/env python3
# TODO: replace prints with logging.
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
import fire
from pathlib import Path
from multiprocessing import cpu_count
from dataclasses import dataclass
from clustering_base import sim_to_dist, process_clustering_result, clustering_result, read_similarity_mat

def affinity_clustering(similarities, output, *args, **kwargs):
    subreddits, mat = read_similarity_mat(similarities)
    clustering = _affinity_clustering(mat, *args, **kwargs)
    cluster_data = process_clustering_result(clustering, subreddits)
    cluster_data['algorithm'] = 'affinity'
    return(cluster_data)

def _affinity_clustering(mat, subreddits, output, damping=0.9, max_iter=100000, convergence_iter=30, preference_quantile=0.5, random_state=1968, verbose=True):
    '''
    similarities: matrix of similarity scores
    preference_quantile: parameter controlling how many clusters to make. higher values = more clusters. 0.85 is a good value with 3000 subreddits.
    damping: parameter controlling how iterations are merged. Higher values make convergence faster and more dependable. 0.85 is a good value for the 10000 subreddits by author. 
    '''
    print(f"damping:{damping}; convergenceIter:{convergence_iter}; preferenceQuantile:{preference_quantile}")

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

    cluster_data = process_clustering_result(clustering, subreddits)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cluster_data.to_feather(output)
    print(f"saved {output}")
    return clustering



if __name__ == "__main__":
    fire.Fire(affinity_clustering)
