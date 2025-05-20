#!/usr/bin/bash

# change according to your computing resources, of course
#SBATCH --account=ucb338_asc3
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=3
#SBATCH --time=6:00:00
#SBATCH --job-name=beam_sim_run
#SBATCH --partition=amilan
#SBATCH --output=beam_sim_run.%j.out
#SBATCH --constraint=ib 
#SBATCH --array=0-7

# activate conda environment
source activate base
conda activate beam_sim_env

# Start a new simulation.
# Note that the task ID array above must match the length of the energy list provided. This will run the simulation once for each array value,
# taking the value of the energy array at that index. 
# Implemented atmosphere types: 'T900_g5.0', 'T1400_g4.0', 'T900_g4.0', 'T1400_g5.0', 'T2000_g5.0', 'T482_g4.7', 'Jupiter'
atm_type=Jupiter
energies=(0.1 0.5 1 5 10 50 100 500)
logfile=beam_sim_run.$SLURM_JOB_ID.out
sonora_filepath=../../atm_data # must have saved atmospheric density profile data here
python ../../beam_simulation_clean.py --Ne0 1000 --Nsteps 20000 --e0 ${energies[$SLURM_ARRAY_TASK_ID]} --atm_type $atm_type --sonora_filepath $sonora_filepath --d --s --logfile $logfile --cos_theta 1




