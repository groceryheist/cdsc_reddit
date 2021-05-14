import fire
from affinity_clustering import affinity_clustering_result, affinity_job, affinity_grid_sweep
from grid_sweep import grid_sweep
from lsi_base import lsi_result_mixin, lsi_grid_sweep, lsi_mixin
from dataclasses import dataclass

@dataclass
class affinity_clustering_result_lsi(affinity_clustering_result, lsi_result_mixin):
    pass


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
                         [self.lsi_dim],
                         *args,
                         **kwargs)

    def namer(self, *args, **kwargs):
        s = affinity_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s
                         
def run_affinity_lsi_grid_sweep(savefile, inpath, outpath, dampings=[0.8], max_iters=[3000], convergence_iters=[30], preference_quantiles=[0.5], lsi_dimensions='all',n_cores=30):
    """Run affinity clustering once or more with different parameters.
    
    Usage:
    affinity_clustering.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH --max_iters=<csv> --dampings=<csv> --preference_quantiles=<csv> --lsi_dimensions: either "all" or one or more available lsi similarity dimensions at INPATH.

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to folder containing feather files with LSI similarity labeled matrices of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    dampings:one or more numbers in [0.5, 1). damping parameter in affinity propagatin clustering. 
    preference_quantiles:one or more numbers in (0,1) for selecting the 'preference' parameter.
    convergence_iters:one or more integers of number of iterations without improvement before stopping.
    max_iters: one or more numbers of different maximum interations.
    lsi_dimensions: either "all" or one or more available lsi similarity dimensions at INPATH.
    """
    
    obj = affinity_lsi_grid_sweep(inpath,
                            lsi_dimensions,
                            outpath,
                            map(float,dampings),
                            map(int,max_iters),
                            map(int,convergence_iters),
                            map(float,preference_quantiles))

    obj.run(n_cores)
    obj.save(savefile)

if __name__ == "__main__":
    fire.Fire(run_affinity_lsi_grid_sweep)
