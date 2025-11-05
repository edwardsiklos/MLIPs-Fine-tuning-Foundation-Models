#!/bin/bash 
#$ -cwd
#$ -pe smp 16
#$ -l s_rt=12:00:00
#$ -j y
#$ -o $JOB_ID.log

# sets the working directory to the one submitted from
# pe smp = number of cores. Our 2 nodes, accessed via omega, have 48 cores each.
# s_rt = soft run-time limit
# -j y merge stderr and stdout 
# -o choose $JOB_ID.log as the outfile

# set PATH, LD_LIBRARY_PATH etc. for the intel libraries used to compile lammps
module load intel/2019
module load mpi/mpich-x86_64
export DIR=$(pwd)

#---------------------- Bits you may need to change ---------------------------------

# all input files and the lammps executable should be in the project directory

restart=data # choose either data or continuation as starting structure for run
lmp_in=$DIR/in_nvt # lammps infile, for constant 'NVT' ensemble MD
#lmp_exec=$DIR/lmp_mpi # LAMMPS executable binary file
lmp_exec=$DIR/lmp_cpu_mpich_quip # LAMMPS executable binary file

# Starting structure - provided by qsub -v STRUCTURE=... in the run script
s=$STRUCTURE

# Run Script Parameters
# This is the standard protocol for amorphous carbon modelling as outlined in GAP17 Methodology
t_randomize=9000 
randomize_timesteps=3000   
t_melt=5000          
melt_timesteps=3000  
t_cool=300            
cool_timesteps=500    
t_anneal=300          
anneal_timesteps=3000  

# Name of the run directory that gets created - based on simulation parameters captured in unique_key
unique_key=$UNIQUE_KEY

rundir="${unique_key}_${1}"              

# e.g. C_GAP17_NVT_216_1.5gcm_1
#-----------------------------------------------------------------------------------

# this takes care of copying back your data from the compute node when the job finishes
function Cleanup ()
{
    trap "" SIGUSR1 EXIT # Disable trap now we're in it
    # Clean up task
    rsync -rltq $TMPDIR/ $DIR/
    exit 0
}
trap Cleanup SIGUSR1 EXIT # Enable trap

# some environment variables for parallelisation and memory usageage
# LAMMPS mainly uses MPI parallelisation (at least with QUIP), so we 
# 'turn off' the OpenMP parallelisation by setting the number of threads
# to 1 (also for the intel Math Kernel Library (MKL) that handles matrix
# operations etc.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NMPI=$(expr $NSLOTS / $OMP_NUM_THREADS )
export GFORTRAN_UNBUFFERED_ALL=y
ulimit -s unlimited

mkdir -p ${rundir}/NVT
mkdir -p ${rundir}/restart
cp $s ${rundir}

# some lammps settings (units, setting up the potential)
system=C
model=gap
units=metal
## the pairstyle below is currently re-defined in the in_nvt input, so these will do nothing
pot=$DIR/carbon.xml # filename for the GAP potential that you use for the simulation
# make sure that the 'sparseX' file(s) are in the same directory as the .xml
pair_style=quip
pot=$(readlink -e 'carbon.xml')
pair_coeff="* * ${pot} \"\" 6"

INFILE="${rundir}"
rsync -rltq $INFILE $TMPDIR/
cd $TMPDIR  # use the temporary directory local to the compute node. This avoids
# writing output over the network filesystem, which is slow for you and slows down
# the NFS for everyone else (especially as jobs get larger)

mpirun -np $NMPI $lmp_exec -in ${lmp_in} \
   -var rand $(od -vAn -N4 -td4 < /dev/urandom | sed "s/-//") \
   -var system ${system} \
   -var units ${units} \
   -var pair_style "${pair_style}" \
   -var pair_coeff "${pair_coeff}" \
\
  -var t_randomize ${t_randomize} \
  -var randomize_timesteps ${randomize_timesteps} \
  -var t_melt ${t_melt} \
  -var melt_timesteps ${melt_timesteps} \
  -var t_cool ${t_cool} \
  -var cool_timesteps ${cool_timesteps} \
  -var t_anneal ${t_anneal} \
  -var anneal_timesteps ${anneal_timesteps} \
\
   -var model ${model} \
   -var rundir ${rundir} \
   -var restart_from ${restart} \
   -var data_file ${rundir}/$(basename $s) &

pid=$! # copy back job data every 10 seconds while we wait for it to finish
while kill -0 $pid 2> /dev/null; do
    sleep 300
    rsync -rltq $TMPDIR/ $DIR/
done
wait $pid

cd $DIR
mv $JOB_ID.log $DIR/$rundir/