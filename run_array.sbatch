#!/bin/bash
## tf reddit comments
#SBATCH --job-name="wikia ecology; fit var models"
## Allocation Definition
#SBATCH --account=comdata-ckpt
#SBATCH --partition=ckpt
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
#SBATCH --chdir /gscratch/comdata/users/nathante/wikia_ecology
#SBATCH --output=var_jobs/%A_%a.out
#SBATCH --error=var_jobs/%A_%a.out
TASK_NUM=$(( SLURM_ARRAY_TASK_ID + $1))
TASK_CALL=$(sed -n ${TASK_NUM}p ./var_jobs.sh)
${TASK_CALL}
