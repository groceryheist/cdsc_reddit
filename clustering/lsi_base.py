from clustering_base import clustering_job, clustering_result
from grid_sweep import grid_sweep, twoway_grid_sweep
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

class lsi_mixin():
    def set_lsi_dims(self, lsi_dims):
        self.lsi_dims = lsi_dims

@dataclass
class lsi_result_mixin:
    lsi_dimensions:int

class lsi_grid_sweep(grid_sweep):
    def __init__(self, jobtype, subsweep, inpath, lsi_dimensions, outpath, *args, **kwargs):
        self.jobtype = jobtype
        self.subsweep = subsweep
        inpath = Path(inpath)
        if lsi_dimensions == 'all':
            lsi_paths = list(inpath.glob("*.feather"))
        else:
            lsi_paths = [inpath / (str(dim) + '.feather') for dim in lsi_dimensions]

        print(lsi_paths)
        lsi_nums = [int(p.stem) for p in lsi_paths]
        self.hasrun = False
        self.subgrids = [self.subsweep(lsi_path, outpath,  lsi_dim, *args, **kwargs) for lsi_dim, lsi_path in zip(lsi_nums, lsi_paths)]
        self.jobs = list(chain(*map(lambda gs: gs.jobs, self.subgrids)))

class twoway_lsi_grid_sweep(twoway_grid_sweep):
    def __init__(self, jobtype, subsweep, inpath, lsi_dimensions, outpath, args1, args2, save_step1):
        self.jobtype = jobtype
        self.subsweep = subsweep
        inpath = Path(inpath)
        if lsi_dimensions == 'all':
            lsi_paths = list(inpath.glob("*.feather"))
        else:
            lsi_paths = [inpath / (str(dim) + '.feather') for dim in lsi_dimensions]

        lsi_nums = [int(p.stem) for p in lsi_paths]
        self.hasrun = False
        self.subgrids = [self.subsweep(lsi_path, outpath, lsi_dim, args1, args2, save_step1) for lsi_dim, lsi_path in zip(lsi_nums, lsi_paths)]
        self.jobs = list(chain(*map(lambda gs: gs.jobs, self.subgrids)))
