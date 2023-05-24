#!/bin/bash
#SBATCH --job-name="simulate measurement error models"
## Allocation Definition
#SBATCH --account=comdata
#SBATCH --partition=compute-bigmem
## Resources
#SBATCH --nodes=1    
## Walltime (4 hours)
#SBATCH --time=4:00:00
## Memory per node
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --chdir /gscratch/comdata/users/nathante/partitioning_reddit/dataverse/cdsc_reddit/ngrams/
#SBATCH --output=sbatch_log/%A_%a.out
#SBATCH --error=sbatch_log/%A_%a.err
echo "$@"
"$@"
