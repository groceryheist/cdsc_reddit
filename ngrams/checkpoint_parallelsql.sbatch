#!/bin/bash
## parallel_sql_job.sh
#SBATCH --job-name=tf_subreddit_comments
## Allocation Definition
#SBATCH --account=comdata-ckpt
#SBATCH --partition=ckpt
## Resources
## Nodes. This should always be 1 for parallel-sql.
#SBATCH --nodes=1    
## Walltime (12 hours)
#SBATCH --time=12:00:00
## Memory per node
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH -D /gscratch/comdata/users/nathante/cdsc-reddit
source ./bin/activate
module load parallel_sql
echo $(which perl)
conda list pyarrow
which python3

parallel-sql --sql -a parallel --exit-on-term --jobs 4
