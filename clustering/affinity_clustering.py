from sklearn.metrics import silhouette_score
from sklearn.cluster import AffinityPropagation
from functools import partial
from dataclasses import dataclass
from clustering_base import sim_to_dist, process_clustering_result, clustering_result, read_similarity_mat
from clustering_base import lsi_result_mixin, lsi_mixin, clustering_job, grid_sweep, lsi_grid_sweep
from multiprocessing  import Pool, cpu_count, Array, Process
from pathlib import Path
from itertools import product, starmap
import numpy as np
import pandas as pd
import fire
import sys

# silhouette is the only one that doesn't need the feature matrix. So it's probably the only one that's worth trying. 
@dataclass
class affinity_clustering_result(clustering_result):
    damping:float
    convergence_iter:int
    preference_quantile:float
    preference:float
    max_iter:int

@dataclass
class affinity_clustering_result_lsi(affinity_clustering_result, lsi_result_mixin):
    pass

class affinity_job(clustering_job):
    def __init__(self, infile, outpath, name, damping=0.9, max_iter=100000, convergence_iter=30, preference_quantile=0.5, random_state=1968, verbose=True):
        super().__init__(infile,
                         outpath,
                         name,
                         call=self._affinity_clustering,
                         preference_quantile=preference_quantile,
                         damping=damping,
                         max_iter=max_iter,
                         convergence_iter=convergence_iter,
                         random_state=1968,
                         verbose=verbose)
        self.damping=damping
        self.max_iter=max_iter
        self.convergence_iter=convergence_iter
        self.preference_quantile=preference_quantile

    def _affinity_clustering(self, mat, preference_quantile, *args, **kwargs):
        mat = 1-mat
        preference = np.quantile(mat, preference_quantile)
        self.preference = preference
        print(f"preference is {preference}")
        print("data loaded")
        sys.stdout.flush()
        clustering = AffinityPropagation(*args,
                                         preference=preference,
                                         affinity='precomputed',
                                         copy=False,
                                         **kwargs).fit(mat)
        return clustering

    def get_info(self):
        result = super().get_info()
        self.result=affinity_clustering_result(**result.__dict__,
                                               damping=self.damping,
                                               max_iter=self.max_iter,
                                               convergence_iter=self.convergence_iter,
                                               preference_quantile=self.preference_quantile,
                                               preference=self.preference)

        return self.result

class affinity_lsi_job(affinity_job, lsi_mixin):
    def __init__(self, infile, outpath, name, lsi_dims, *args, **kwargs):
        super().__init__(infile,
                         outpath,
                         name,
                         *args,
                         **kwargs)
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        result = super().get_info()
        self.result = affinity_clustering_result_lsi(**result.__dict__,
                                                     lsi_dimensions=self.lsi_dims)
        return self.result

class affinity_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 *args,
                 **kwargs):

        super().__init__(affinity_job,
                         _afffinity_grid_sweep,
                         inpath,
                         outpath,
                         self.namer,
                         *args,
                         **kwargs)
    def namer(self,
              damping,
              max_iter,
              convergence_iter,
              preference_quantile):

        return f"damp-{damping}_maxit-{max_iter}_convit-{convergence_iter}_prefq-{preference_quantile}"

class _affinity_lsi_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 *args,
                 **kwargs):
        self.lsi_dim = lsi_dim
        self.jobtype = affinity_lsi_job
        super().__init__(self.jobtype,
                         inpath,
                         outpath,
                         self.namer,
                         self.lsi_dim,
                         *args,
                         **kwargs)

    def namer(self, *args, **kwargs):
        s = affinity_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

class affinity_lsi_grid_sweep(lsi_grid_sweep):
    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 dampings=[0.9],
                 max_iters=[10000],
                 convergence_iters=[30],
                 preference_quantiles=[0.5]):

        super().__init__(affinity_lsi_job,
                         _affinity_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         dampings,
                         max_iters,
                         convergence_iters,
                         preference_quantiles)
    
                         
    
def test_select_affinity_clustering():
    # select_hdbscan_clustering("/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_30k_LSI",
    #                           "test_hdbscan_author30k",
    #                           min_cluster_sizes=[2],
    #                           min_samples=[1,2],
    #                           cluster_selection_epsilons=[0,0.05,0.1,0.15],
    #                           cluster_selection_methods=['eom','leaf'],
    #                           lsi_dimensions='all')
    inpath = "/gscratch/comdata/output/reddit_similarity/subreddit_comment_authors-tf_10k_LSI/"
    outpath = "test_affinity";
    dampings=[0.8,0.9]
    max_iters=[100000]
    convergence_iters=[15]
    preference_quantiles=[0.5,0.7]
    
    gs = affinity_lsi_grid_sweep(inpath, 'all', outpath, dampings, max_iters, convergence_iters, preference_quantiles)
    gs.run(20)
    gs.save("test_affinity/lsi_sweep.csv")


if __name__ == "__main__":
    fire.Fire{'grid_sweep':affinity_grid_sweep,
              'grid_sweep_lsi':affinity_lsi_grid_sweep
              'cluster':affinity_job,
              'cluster_lsi':affinity_lsi_job}
