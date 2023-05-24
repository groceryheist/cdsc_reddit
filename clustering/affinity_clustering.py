from sklearn.cluster import AffinityPropagation
from dataclasses import dataclass
from clustering_base import clustering_result, clustering_job
from grid_sweep import grid_sweep
from pathlib import Path
from itertools import product, starmap
import fire
import sys
import numpy as np

@dataclass
class affinity_clustering_result(clustering_result):
    damping:float
    convergence_iter:int
    preference_quantile:float
    preference:float
    max_iter:int

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

def run_affinity_grid_sweep(savefile, inpath, outpath, dampings=[0.8], max_iters=[3000], convergence_iters=[30], preference_quantiles=[0.5],n_cores=10):
    """Run affinity clustering once or more with different parameters.
    
    Usage:
    affinity_clustering.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --max_iters=<csv> --dampings=<csv> --preference_quantiles=<csv>

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to feather data containing a labeled matrix of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    dampings:one or more numbers in [0.5, 1). damping parameter in affinity propagatin clustering. 
    preference_quantiles:one or more numbers in (0,1) for selecting the 'preference' parameter.
    convergence_iters:one or more integers of number of iterations without improvement before stopping.
    max_iters: one or more numbers of different maximum interations.
    """
    obj = affinity_grid_sweep(inpath,
                         outpath,
                         map(float,dampings),
                         map(int,max_iters),
                         map(int,convergence_iters),
                         map(float,preference_quantiles))
    obj.run(n_cores)
    obj.save(savefile)
    
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
    fire.Fire(run_affinity_grid_sweep)
