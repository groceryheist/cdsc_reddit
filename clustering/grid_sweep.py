from pathlib import Path
from multiprocessing import Pool, cpu_count
from itertools import product, chain
import pandas as pd

class grid_sweep:
    def __init__(self, jobtype, inpath, outpath, namer, *args):
        self.jobtype = jobtype
        self.namer = namer
        print(*args)
        grid = list(product(*args))
        inpath = Path(inpath)
        outpath = Path(outpath)
        self.hasrun = False
        self.grid = [(inpath,outpath,namer(*g)) + g for g in grid]
        self.jobs = [jobtype(*g) for g in self.grid]

    def run(self, cores=20):
        if cores is not None and cores > 1:
            with Pool(cores) as pool:
                infos = pool.map(self.jobtype.get_info, self.jobs)
        else:
            infos = map(self.jobtype.get_info, self.jobs)

        self.infos = pd.DataFrame(infos)
        self.hasrun = True

    def save(self, outcsv):
        if not self.hasrun:
            self.run()
        outcsv = Path(outcsv)
        outcsv.parent.mkdir(parents=True, exist_ok=True)
        self.infos.to_csv(outcsv)
