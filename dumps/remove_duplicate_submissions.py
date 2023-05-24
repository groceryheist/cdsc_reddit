from pathlib import Path
from itertools import chain, groupby

dumpdir = Path("/gscratch/comdata/raw_data/reddit_dumps/submissions")

zst_files = dumpdir.glob("*.zst")
bz2_files = dumpdir.glob("*.bz2")
xz_files = dumpdir.glob("*.xz")
all_files = sorted(list(chain(zst_files, bz2_files, xz_files)))
groups = groupby(all_files, key = lambda p: p.stem)

kept_paths = []
removed_paths = []

priority = ['.zst','.xz','.bz2']

for stem, files in groups:
    keep_file = None
    remove_files = []
    for f in files:
        if keep_file is None:
            keep_file = f
        elif priority.index(keep_file.suffix) > priority.index(f.suffix):
            remove_files.append(keep_file)
            keep_file = f
        else:
            remove_files.append(f)
    kept_paths.append(keep_file)
    removed_paths.extend(remove_files)

(dumpdir / "to_remove").mkdir()

for f in removed_paths:
    f.rename(f.parent / "to_remove" / f.name)
