#!/usr/bin/env python3
# run from a build_machine

import requests
from os import path
import hashlib

shasums1 = requests.get("https://files.pushshift.io/reddit/comments/sha256sum.txt").text

shasums = shasums1 
dumpdir = "/gscratch/comdata/raw_data/reddit_dumps/comments"

for l in shasums.strip().split('\n'):
    sha256_hash = hashlib.sha256()
    parts = l.split(' ')

    correct_sha256 = parts[0]
    filename = parts[-1]
    print(f"checking {filename}")
    fpath = path.join(dumpdir,filename)
    if path.isfile(fpath):
        with open(fpath,'rb') as f:
            for byte_block in iter(lambda: f.read(4096),b""):
                sha256_hash.update(byte_block)

        if sha256_hash.hexdigest() == correct_sha256:
            print(f"{filename} checks out")
        else:
            print(f"ERROR! {filename} has the wrong hash. Redownload and recheck!")
    else:
        print(f"Skipping {filename} as it doesn't exist")

