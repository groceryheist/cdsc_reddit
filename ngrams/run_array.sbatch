#!/bin/bash
#SBATCH --job-name=reddit_comment_term_frequencies
#SBATCH --account=comdata
#SBATCH --partition=compute-bigmem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=9g
#SBATCH --ntasks=1
#SBATCH --export=ALL
#SBATCH --time=48:00:00
#SBATCH --chdir=/gscratch/comdata/users/nathante/partitioning_reddit/dataverse/cdsc_reddit/ngrams
#SBATCH --error="sbatch_log/%A_%a.out"
#SBATCH --output="sbatch_log/%A_%a.out"

TASK_NUM=$(($SLURM_ARRAY_TASK_ID + $1))
TASK_CALL=$(sed -n ${TASK_NUM}p $2)
${TASK_CALL}

