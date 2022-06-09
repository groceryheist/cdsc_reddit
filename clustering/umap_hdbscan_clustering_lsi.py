from umap_hdbscan_clustering import umap_hdbscan_job, umap_hdbscan_grid_sweep, umap_hdbscan_clustering_result
from lsi_base import twoway_lsi_grid_sweep, lsi_mixin, lsi_result_mixin
from grid_sweep import twoway_grid_sweep
import fire
from dataclasses import dataclass

@dataclass
class umap_hdbscan_clustering_result_lsi(umap_hdbscan_clustering_result, lsi_result_mixin):
    pass 

class umap_hdbscan_lsi_job(umap_hdbscan_job, lsi_mixin):
    def __init__(self, infile, outpath, name, umap_args, hdbscan_args, lsi_dims):
        super().__init__(
            infile,
            outpath,
            name,
            umap_args,
            hdbscan_args
        )
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        partial_result = super().get_info()
        self.result = umap_hdbscan_clustering_result_lsi(**partial_result.__dict__,
                                                         lsi_dimensions=self.lsi_dims)
        return self.result

class umap_hdbscan_lsi_grid_sweep(twoway_lsi_grid_sweep):
    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 umap_args,
                 hdbscan_args
                 ):

        super().__init__(umap_hdbscan_lsi_job,
                         _umap_hdbscan_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         umap_args,
                         hdbscan_args
                         )
        


class _umap_hdbscan_lsi_grid_sweep(twoway_grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 umap_args,
                 hdbscan_args,
                 ):

        self.lsi_dim = lsi_dim
        self.jobtype = umap_hdbscan_lsi_job
        super().__init__(self.jobtype, inpath, outpath, self.namer, umap_args, hdbscan_args, lsi_dim)


    def namer(self, *args, **kwargs):
        s = umap_hdbscan_grid_sweep.namer(self, *args, **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

def run_umap_hdbscan_lsi_grid_sweep(savefile, inpath, outpath, n_neighbors = [15], n_components=[2], learning_rate=[1], min_dist=[1], local_connectivity=[1], 
                                densmap=[False],
                                    min_cluster_sizes=[2], min_samples=[1], cluster_selection_epsilons=[0], cluster_selection_methods=['eom'], lsi_dimensions='all'):
    """Run hdbscan clustering once or more with different parameters.
    
    Usage:
    hdbscan_clustering_lsi --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --min_cluster_sizes=<csv> --min_samples=<csv> --cluster_selection_epsilons=<csv> --cluster_selection_methods=[eom]> --lsi_dimensions: either "all" or one or more available lsi similarity dimensions at INPATH.

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to folder containing feather files with LSI similarity labeled matrices of subreddit similarities.
    outpath: path to output fit clusterings.
    min_cluster_sizes: one or more integers indicating the minumum cluster size
    min_samples: one ore more integers indicating the minimum number of samples used in the algorithm
    cluster_selection_epsilons: one or more similarity thresholds for transition from dbscan to hdbscan
    cluster_selection_methods: one or more of "eom" or "leaf" eom gives larger clusters. 
    lsi_dimensions: either "all" or one or more available lsi similarity dimensions at INPATH.
    """    


    umap_args = {'n_neighbors':list(map(int, n_neighbors)),
                 'learning_rate':list(map(float,learning_rate)),
                 'min_dist':list(map(float,min_dist)),
                 'local_connectivity':list(map(int,local_connectivity)),
                 'n_components':list(map(int, n_components)),
                 'densmap':list(map(bool,densmap))
                 }

    hdbscan_args = {'min_cluster_size':list(map(int,min_cluster_sizes)),
                    'min_samples':list(map(int,min_samples)),
                    'cluster_selection_epsilon':list(map(float,cluster_selection_epsilons)),
                    'cluster_selection_method':cluster_selection_methods}

    obj = umap_hdbscan_lsi_grid_sweep(inpath,
                                      lsi_dimensions,
                                      outpath,
                                      umap_args,
                                      hdbscan_args
                                      )
                                 

    obj.run(10)
    obj.save(savefile)


if __name__ == "__main__":
    fire.Fire(run_umap_hdbscan_lsi_grid_sweep)
