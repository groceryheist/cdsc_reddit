from subprocess import Popen, PIPE
import re
from collections import defaultdict
from os import path
import glob

def find_dumps(dumpdir, base_pattern):

    files = glob.glob(path.join(dumpdir,base_pattern))

    # build a dictionary of possible extensions for each dump
    dumpext = defaultdict(list)
    for fpath in files:
        fname, ext = path.splitext(fpath)
        dumpext[fname].append(ext)

    ext_priority = ['.zst','.xz','.bz2']

    for base, exts in dumpext.items():
        ext = [ext for ext in ext_priority if ext in exts][0]
        yield base + ext

def open_fileset(files):
    for fh in files:
        print(fh)
        lines = open_input_file(fh)
        for line in lines:
            yield line

def open_input_file(input_filename):
    if re.match(r'.*\.7z$', input_filename):
        cmd = ["7za", "x", "-so", input_filename, '*'] 
    elif re.match(r'.*\.gz$', input_filename):
        cmd = ["zcat", input_filename] 
    elif re.match(r'.*\.bz2$', input_filename):
        cmd = ["bzcat", "-dk", input_filename] 
    elif re.match(r'.*\.bz', input_filename):
        cmd = ["bzcat", "-dk", input_filename] 
    elif re.match(r'.*\.xz', input_filename):
        cmd = ["xzcat",'-dk', '-T 20',input_filename]
    elif re.match(r'.*\.zst',input_filename):
        cmd = ['zstd','-dck', input_filename]
    try:
        input_file = Popen(cmd, stdout=PIPE).stdout
    except NameError as e:
        print(e)
        input_file = open(input_filename, 'r')
    return input_file

