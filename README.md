# Overview
This repository contains the code used to produce results and plots for Zuckerman et al. 2025. Though the simulation code is provided here, it is the parameterization of the interaction rates which should be most useful to other users. 

# Summary of Files
### beam_simulation_clean.py
Main code of simulation of aurorally precipitating energetic electrons in H2 dominated substellar atmospheres. 

### utils_clean.py
Functions called by `beam_simulation_clean.py` and `make_Zuckerman2025_paper_plots.py`.

### run_beam_sim.sh
Example jobscript to run `beam_simulation_clean.py`. 

### make_Zuckerman2025_paper_plots.py
Script to produce all plots in Zuckerman et. al. 2025. Filepaths to neccessary simulation runs must be provided.

### beam_sim_environment.yml
Environment file. Can be created using "conda env create -f beam_sim_environment.yml"



