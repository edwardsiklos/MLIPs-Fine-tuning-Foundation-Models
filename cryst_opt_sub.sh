#!/bin/bash
#$ -cwd
#$ -pe smp 4
#$ -l s_rt=02:00:00
#$ -j y
#$ -o $JOB_ID_cryst_opt.log

# 1) Initialise conda
source /u/vld/scat9451/miniconda3/etc/profile.d/conda.sh

# 2) Activate the quippy environment
conda activate quippy

# NOTE: 
# This env is very sensitive and must have all numpy<2
# -------------------------------------------------------------------------------------------------------------
# Instructions for creating suitable env:
# conda create -n quippy python=3.10 -y
# conda activate quippy
# pip install "numpy<2" quippy-ase mace-torch graph-pes
# conda install --strict-channel-priority -c https://conda.ovito.org -c conda-forge "numpy<2" ovito=3.14.1 -y
# -------------------------------------------------------------------------------------------------------------

# 3) Optional: use all reserved cores for threaded libs (PyTorch/BLAS)
export OMP_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=1 

# OPENBLAS_NUM_THREADS=1 used to prevent: 
# OpenBLAS Warning : Detect OpenMP Loop and this application may hang. Please rebuild the library with USE_OPENMP=1 option.
# This stops .xml file multithreading 

# 4) Run your script (assuming it's in the submission directory)
python Crystalline_Optimization.py
