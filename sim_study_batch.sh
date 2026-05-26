#!/bin/bash -l

#$ -P modislc
#$ -N sim_study_batch
#$ -pe omp 1
#$ -t 1-800
#$ -e /projectnb/modislc/users/danc/SOFM/logs/
#$ -o /projectnb/modislc/users/danc/SOFM/logs/

module load miniconda
conda activate spatcca
python sim_study_batch.py --task_id $SGE_TASK_ID