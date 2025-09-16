#!/usr/bin/bash

#SBATCH --account=your_account
#SBATCH --ntasks=64
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=beam_sim_run
#SBATCH --partition=your_partition
#SBATCH --output=beam_sim_run.%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email
#SBATCH --qos=normal
#SBATCH --array=0-7

# activate conda environment (will vary depending on where you're running this of course)
module load anaconda
conda activate pymc_env

# enable buffer
export PYTHONUNBUFFERED=TRUE

# Start a new simulation.
# Note that the task ID array above must match the length of the energy list provided. This will run the simulation once for each array value,
# taking the value of the energy array at that index. 
atm_type=T2000_g5.0 # See implemented types in beam_simulation_clean.py
energies=(500 100 50 10 5 1 0.5 0.1)
logfile=beam_sim_run.${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
sonora_filepath=../../Sonora_data # atmospheric profile output from Picaso/Sonora model, available upon request
python ../../beam_simulation_clean.py --Ne0 1000 --e0 ${energies[$SLURM_ARRAY_TASK_ID]} --atm_type $atm_type --sonora_filepath $sonora_filepath --d --s --min --logfile $logfile --cos_theta 1








