from hdbscan_clustering import hdbscan_job, hdbscan_grid_sweep, hdbscan_clustering_result
from lsi_base import lsi_grid_sweep, lsi_mixin, lsi_result_mixin
from grid_sweep import grid_sweep
import fire
from dataclasses import dataclass

@dataclass
class hdbscan_clustering_result_lsi(hdbscan_clustering_result, lsi_result_mixin):
    pass 

class hdbscan_lsi_job(hdbscan_job, lsi_mixin):
    def __init__(self, infile, outpath, name, lsi_dims, *args, **kwargs):
        super().__init__(
                         infile,
                         outpath,
                         name,
                         *args,
                         **kwargs)
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        partial_result = super().get_info()
        self.result = hdbscan_clustering_result_lsi(**partial_result.__dict__,
                                                    lsi_dimensions=self.lsi_dims)
        return self.result

class hdbscan_lsi_grid_sweep(lsi_grid_sweep):
    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 min_cluster_sizes,
                 min_samples,
                 cluster_selection_epsilons,
                 cluster_selection_methods
                 ):

        super().__init__(hdbscan_lsi_job,
                         _hdbscan_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         min_cluster_sizes,
                         min_samples,
                         cluster_selection_epsilons,
                         cluster_selection_methods)
        


class _hdbscan_lsi_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 *args,
                 **kwargs):
        print(args)
        print(kwargs)

        self.lsi_dim = lsi_dim
        self.jobtype = hdbscan_lsi_job
        super().__init__(self.jobtype, inpath, outpath, self.namer, [self.lsi_dim], *args, **kwargs)


    def namer(self, *args, **kwargs):
        s = hdbscan_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

def run_hdbscan_lsi_grid_sweep(savefile, inpath, outpath,  min_cluster_sizes=[2], min_samples=[1], cluster_selection_epsilons=[0], cluster_selection_methods=[1],lsi_dimensions='all'):
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

    obj = hdbscan_lsi_grid_sweep(inpath,
                                 lsi_dimensions,
                                 outpath,
                                 list(map(int,min_cluster_sizes)),
                                 list(map(int,min_samples)),
                                 list(map(float,cluster_selection_epsilons)),
                                 cluster_selection_methods)
                                 

    obj.run(10)
    obj.save(savefile)


if __name__ == "__main__":
    fire.Fire(run_hdbscan_lsi_grid_sweep)
