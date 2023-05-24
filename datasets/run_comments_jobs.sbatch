#!/bin/bash
## tf reddit comments
#SBATCH --job-name="cdsc_reddit; parse comment dumps"
## Allocation Definition
#SBATCH --account=comdata
#SBATCH --partition=compute-bigmem
## Resources
## Nodes. This should always be 1 for parallel-sql.
#SBATCH --nodes=1    
## Walltime (12 hours)
#SBATCH --time=24:00:00
## Memory per node
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH 
#SBATCH --chdir /gscratch/comdata/users/nathante/partitioning_reddit/dataverse/cdsc_reddit/datasets
#SBATCH --output=comments_jobs/%A_%a.out
#SBATCH --error=comments_jobs/%A_%a.out
. /opt/ohpc/admin/lmod/lmod/init/profile
source ~/.bashrc
TASK_NUM=$(( SLURM_ARRAY_TASK_ID + $1))
TASK_CALL=$(sed -n ${TASK_NUM}p ./comments_task_list.sh)
${TASK_CALL}
