import fire
from dataclasses import dataclass
from kmeans_clustering import kmeans_job, kmeans_clustering_result, kmeans_grid_sweep
from lsi_base import lsi_mixin, lsi_result_mixin, lsi_grid_sweep
from grid_sweep import grid_sweep

@dataclass
class kmeans_clustering_result_lsi(kmeans_clustering_result, lsi_result_mixin):
    pass

class kmeans_lsi_job(kmeans_job, lsi_mixin):
    def __init__(self, infile, outpath, name, lsi_dims, *args, **kwargs):
        super().__init__(infile,
                         outpath,
                         name,
                         *args,
                         **kwargs)
        super().set_lsi_dims(lsi_dims)

    def get_info(self):
        result = super().get_info()
        self.result = kmeans_clustering_result_lsi(**result.__dict__,
                                                   lsi_dimensions=self.lsi_dims)
        return self.result

class _kmeans_lsi_grid_sweep(grid_sweep):
    def __init__(self,
                 inpath,
                 outpath,
                 lsi_dim,
                 *args,
                 **kwargs):
        print(args)
        print(kwargs)
        self.lsi_dim = lsi_dim
        self.jobtype = kmeans_lsi_job
        super().__init__(self.jobtype, inpath, outpath, self.namer, self.lsi_dim, *args, **kwargs)

    def namer(self, *args, **kwargs):
        s = kmeans_grid_sweep.namer(self, *args[1:], **kwargs)
        s += f"_lsi-{self.lsi_dim}"
        return s

class kmeans_lsi_grid_sweep(lsi_grid_sweep):

    def __init__(self,
                 inpath,
                 lsi_dims,
                 outpath,
                 n_clusters,
                 n_inits,
                 max_iters
                 ):

        super().__init__(kmeans_lsi_job,
                         _kmeans_lsi_grid_sweep,
                         inpath,
                         lsi_dims,
                         outpath,
                         n_clusters,
                         n_inits,
                         max_iters)

def run_kmeans_lsi_grid_sweep(savefile, inpath, outpath,  n_clusters=[500], n_inits=[1], max_iters=[3000], lsi_dimensions="all"):
    """Run kmeans clustering once or more with different parameters.
    
    Usage:
    kmeans_clustering_lsi.py --savefile=SAVEFILE --inpath=INPATH --outpath=OUTPATH d--lsi_dimensions=<"all"|csv number of LSI dimensions to use> --n_clusters=<csv number of clusters> --n_inits=<csv> --max_iters=<csv>

    Keword arguments:
    savefile: path to save the metadata and diagnostics 
    inpath: path to folder containing feather files with LSI similarity labeled matrices of subreddit similarities.
    outpath: path to output fit kmeans clusterings.
    lsi_dimensions: either "all" or one or more available lsi similarity dimensions at INPATH.
    n_clusters: one or more numbers of kmeans clusters to select.
    n_inits: one or more numbers of different initializations to use for each clustering.
    max_iters: one or more numbers of different maximum interations. 
    """    

    obj = kmeans_lsi_grid_sweep(inpath,
                                lsi_dimensions,
                                outpath,
                                list(map(int,n_clusters)),
                                list(map(int,n_inits)),
                                list(map(int,max_iters))
                                )

    obj.run(1)
    obj.save(savefile)


if __name__ == "__main__":
    fire.Fire(run_kmeans_lsi_grid_sweep)
