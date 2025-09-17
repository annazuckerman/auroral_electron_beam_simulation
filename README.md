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

# Example Usage
The key result of Zuckerman et al. 2025 is the paramterization of the interaction rates. Rather than running a new simulation to study a new atmosphere, we can simply specify a density profile and incident beam energy, and use this paramterization. See Section 3.3 of the paper. We can also calculate volumetric rates after specifying an incident electron beam energy spectrum (see Section 4).

To use this parameterization, we first need to specify a density profile function which returns the number density of H2 (m^-3) as a function of altitude z (m). This can be done either by directly defining the function (for instance as a spline to data or an analytic function), or by providing data from a model like Picaso/Sonora. In the second case, you can construct the density profile function by using the function `utils_clean.construct_profiles()`, specifying your path to the model output. The function expects the model output to be in the form of the files in the `Example_Picaso_data` directory. 

To use the parameterization, simply call the function `utils_clean.calc_q()`. An example call is as follows:\
\
`q = utils_clean.calc_q(z, E_keV, z_min, z_max, get_n_H2, interaction_type)`\
where\
`z` is a numpy array of altitude (m) over which to calculate the interaction rate *q*\
`E_keV` is the energy of the incident electron beam (keV)\
`z_min` is the lower bound over which to calculate *q* (m)\
`z_max` is the upper bound over which to calculate *q* (m)\
`get_n_H2` is a function taking a array of altitude (m) and returning H2 number density (m^-3)\
`interaction_type` is a string representing which interaction we want the rate for. Options are listed in the function `calc_Nevent_over_N()`. 


We can also calculate a volumetric interaction rate `Q` by simply numerically integrating, \
$Q(z) = \int q(\varepsilon_0, z) F(\varepsilon_0) d\varepsilon_0$\
where $F(\varepsilon_0)$ is the incident beam energy spectrum. This is done using the function `utils_clean.calc_Q()`. For example\
\
`Q = utils.calc_Q(F, z, E_eV, z_min, z_max, get_n_H2, interaction_type)`\
where\
`F` is a function taking an array of energy in eV and returning the beam spectrum in electrons/m^2/s/eV\
`z` is the float altitude value (m) at which to calculate *Q*\
`E_eV` is an array of energies (eV) over which to calculate the incident electron beam energy spectrum.\
`z_min` is the lower bound over which to calculate *q* (m)\
`z_max` is the upper bound over which to calculate *q* (m)\
`get_n_H2` is a function taking a array of altitude (m) and returning H2 number density (m^-3)\
`interaction_type` is a string representing which interaction we want the rate for. Options are listed in the function `calc_Nevent_over_N()`. 


