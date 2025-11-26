#!/bin/bash
#$ -cwd
#$ -pe smp 4
#$ -l s_rt=02:00:00
#$ -j y
#$ -o $JOB_ID_cryst_opt.log

# 1) Initialise conda
source /u/vld/scat9451/miniconda3/etc/profile.d/conda.sh

# 2) Activate the graph-pes-mace environment
conda activate graph-pes-mace

# 3) Optional: use all reserved cores for threaded libs (PyTorch/BLAS)
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS

# 4) Run your script (assuming it's in the submission directory)
python Crystalline_Optimization.py
