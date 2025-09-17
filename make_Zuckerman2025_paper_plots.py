'''
Code to reproduce all plots from Zuckerman et. al 2025 (as of this wirting, in prep), "Simulations of Electron Beam Interactions in Brown Dwarf Atmospheres"
'''


import numpy as np
def patch_asscalar(a): # hack because of updated numpydeprecations
    return a.item()
setattr(np, "asscalar", patch_asscalar)
def patch_alen(a): # hack because of updated numpydeprecations
    return a.len()
setattr(np, "alen", patch_alen)
import matplotlib.pyplot as plt
from astropy import units as u
import utils
import importlib
import pandas as pd
from mycolorpy import colorlist as mcp
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import stats 
from scipy.integrate import trapezoid
import argparse
from matplotlib import pyplot as plt, ticker as mticker
import os
c = 2.99e8 * u.m / u.s # speed of light [m/s]
c = c.to(u.m/u.s).value
m = (9.1093837e-31 * u.kg).value # e- mass [kg]
mH2 = 3.347649043E-27 # H2 mass [kg]
k = 1.380649e-23 # bolztman constant [J/k]


# parse input args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--picaso_filepath',
                    dest='picaso_filepath',
                    type=str,
                    help='Relative path to Picaso atmospheric profiles.',
                    required=True)
parser.add_argument('--filepath_Jupiter_galileo',
                    dest='filepath_Jupiter_galileo',
                    type=str,
                    help='Filepath to Jupiter (galileo density) results files.',
                    required=False)
parser.add_argument('--filepath_Jupiter_isothermal',
                    dest='filepath_Jupiter_isothermal',
                    type=str,
                    help='Filepath to Jupiter (isothermally extended density) results files.',
                    required=False)
parser.add_argument('--filepath_Jupiter_hiraki',
                    dest='filepath_Jupiter_hiraki',
                    type=str,
                    help='Filepath to Jupiter (Hiraki densiy) results files.',
                    required=False)
parser.add_argument('--filepath_T900_g5',
                    dest='filepath_T900_g5',
                    type=str,
                    help='Filepath to T900_g5 results files.',
                    required=False)
parser.add_argument('--filepath_T900_g4',
                    dest='filepath_T900_g4',
                    type=str,
                    help='Filepath to T900_g4 results files.',
                    required=False)
parser.add_argument('--filepath_T1400_g5',
                    dest='filepath_T1400_g5',
                    type=str,
                    help='Filepath to T1400_g5 results files.',
                    required=False)
parser.add_argument('--filepath_T1400_g4',
                    dest='filepath_T1400_g4',
                    type=str,
                    help='Filepath to T1400_g4 results files.',
                    required=False)
parser.add_argument('--filepath_T2000_g5',
                    dest='filepath_T2000_g5',
                    type=str,
                    help='Filepath to T2000_g5 results files.',
                    required=False)
parser.add_argument('--filepath_T482_g4point7',
                    dest='filepath_T482_g4point7',
                    type=str,
                    help='Filepath to T482_g4point7 results files.',
                    required=False)
parser.add_argument('--out_filepath',
                    dest='out_filepath',
                    type=str,
                    help='Ouput path to save plots.',
                    required=True)
parser.add_argument('--fig2',
                    dest='fig2',
                    action='store_true',
                    help='Whether to make Figure 2.',
                    default=False)
parser.add_argument('--fig3',
                    dest='fig3',
                    action='store_true',
                    help='Whether to make Figure 3.',
                    default=False)
parser.add_argument('--fig4',
                    dest='fig4',
                    action='store_true',
                    help='Whether to make Figure 4.',
                    default=False)
parser.add_argument('--fig5',
                    dest='fig5',
                    action='store_true',
                    help='Whether to make Figure 5.',
                    default=False)
parser.add_argument('--fig6',
                    dest='fig6',
                    action='store_true',
                    help='Whether to make Figure 6.',
                    default=False)
parser.add_argument('--fig7',
                    dest='fig7',
                    action='store_true',
                    help='Whether to make Figure 7.',
                    default=False)
parser.add_argument('--fig8',
                    dest='fig8',
                    action='store_true',
                    help='Whether to make Figure 8.',
                    default=False)
parser.add_argument('--fig9',
                    dest='fig9',
                    action='store_true',
                    help='Whether to make Figure 9.',
                    default=False)
parser.add_argument('--fig10',
                    dest='fig10',
                    action='store_true',
                    help='Whether to make Figure 10.',
                    default=False)
parser.add_argument('--fig11',
                    dest='fig11',
                    action='store_true',
                    help='Whether to make Figure 11.',
                    default=False)
parser.add_argument('--fig12',
                    dest='fig12',
                    action='store_true',
                    help='Whether to make Figure 12.',
                    default=False)
parser.add_argument('--fig13',
                    dest='fig13',
                    action='store_true',
                    help='Whether to make Figure 13.',
                    default=False)
parser.add_argument('--fig14',
                    dest='fig14',
                    action='store_true',
                    help='Whether to make Figure 14.',
                    default=False)
parser.add_argument('--fig15',
                    dest='fig15',
                    action='store_true',
                    help='Whether to make Figure 15.',
                    default=False)

args = parser.parse_args()

def get_files(filepath, energies):
    '''
    Read in and return results dataframes from filepath
    '''
    print('energies:', energies)
    print('filepath:', filepath)
    dfs = [None] * len(energies) # un-pythonic, but simple
    for item in os.listdir(filepath):
        if os.path.isdir(filepath + '/' + item) and item != '.ipynb_checkpoints':
            for file in os.listdir(filepath + '/' + item):
                if file.startswith('results') and not file.endswith('.gz'):
                    energy = float(file.split('E0=')[1].split('cos')[0])
                    if energy in energies:
                        print('energy:', energy)
                        results_file = filepath + '/'  + str(energy) + 'keV/' + file 
                        print('file:', file)
                        print('results_file:', results_file)
                        idx = np.where(energies == energy)[0][0]
                        dfs[idx] = pd.read_hdf(results_file, 'results')
                        print('Energy=' + str(energy) + 'keV: ' + results_file)
    return dfs

## Figure 1 is a diagram made in PowerPoint

## Make Figure 2 
if args.fig2:
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    print('Making Figure 2')
    print('Jupiter (Hiraki profile) path:', args.filepath_Jupiter_hiraki, energies)
    print('Using files:')
    dfs = get_files(args.filepath_Jupiter_hiraki, energies)
    plt.figure(figsize = [8,6], dpi = 200)
    fs = 16
    cmap = mcp.gen_color(cmap="plasma",n=len(energies)+1)
    colors = cmap
    z_min = 0e3
    z_max = 2400e3
    nbins = 80
    Ne = 1000
    get_nH2 = utils.n_H2_Jupiter_Hiraki

    for i in range(len(energies)):
        if energies[i] in [0.5, 5,50,500]:
            z_ions = dfs[i].loc['Ionization heights [m]'].dropna()
            bins_arr = np.linspace(z_min, z_max, nbins)
            binwidth = bins_arr[1] - bins_arr[0]
            counts, bins = np.histogram(z_ions, bins = bins_arr)
            counts = counts / Ne
            binwidth = (bins[1]-bins[0])
            bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth  
            q = counts/binwidth
            plt.step(q, bins[1:]/1000, color = colors[i], label = str(energies[i]) + ' keV')
            e0 = float(energies[i])*1000
            Hiraki_q = utils.get_Hiraki_parameterization_curve(bincenters, z_min, z_max, e0, get_nH2)
            plt.plot(Hiraki_q, bincenters/1000, color = colors[i], ls = 'dashed')
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.plot([],[],ls = 'dashed', color = 'k', label = 'Hiraki & Tao (2008)')
    ax1.minorticks_on()
    ax1.legend(fontsize = 0.8*fs)
    plt.xlim([1e-7, 1e-1])
    plt.savefig(args.out_filepath + '/Hiraki_profile_sim_vs_Hiraki_parameterization_v3.pdf', format = 'pdf', bbox_inches="tight")    


## Make Figure 3
if args.fig3:
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    print('Making Figure 3')
    print('Jupiter (Galileo profile) path:', args.filepath_Jupiter_galileo)
    print('Using files:')
    dfs = get_files(args.filepath_Jupiter_galileo, energies)
    plt.figure(figsize = [8,6], dpi = 200)
    fs = 16
    energies = [500, 100, 50, 10, 5, 1, 0.5, 0.1]
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    z_min = 150e3
    z_max = 2200e3
    nbins = 80
    Ne = 1000
    picaso_filepath = './Picaso_data'
    atm_type = 'Jupiter'

    get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)

    for i in range(len(energies)):
        z_ions = dfs[i].loc['Ionization heights [m]'].dropna()
        bins_arr = np.linspace(z_min, z_max, nbins)
        binwidth = bins_arr[1] - bins_arr[0]
        counts, bins = np.histogram(z_ions, bins = bins_arr)
        counts = counts / Ne
        binwidth = (bins[1]-bins[0])
        bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth  
        q = counts/binwidth
        plt.step(q, bins[1:]/1000, color = colors[i], label = str(energies[i]) + ' keV')
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_P_H2(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.legend(fontsize = 0.8*fs)
    plt.xlim([1e-7, 2.5e-1])

    plt.savefig(args.out_filepath + '/Jupiter_Galileo_v3.pdf', format = 'pdf', bbox_inches="tight")
    
## Make Figure 4
if args.fig4:
    plt.figure(figsize = [8,6], dpi = 200)
    fs = 16
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    print('Making Figure 4')
    print('Jupiter (Galileo profile) path:', args.filepath_Jupiter_galileo)
    print('Using files:')
    dfs = get_files(args.filepath_Jupiter_galileo, energies)
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    z_min = 150e3
    z_max = 2200e3
    NoverN0_min = 1e-6
    NoverN0_max = 20e0
    nbins = 50
    Ne = 1000
    mH2 = 3.347649043E-27 # H2 mass [kg]
    atm_type = 'Jupiter'
    npoints = 500
    
    picaso_filepath = './Picaso_data'
    get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)
    z_grid = np.linspace(0,z_max,100)
    N_grid = utils.construct_R_grid(z_grid, z_min, z_max, get_n_H2)
    
    for i in range(len(energies)):
        z_ions = dfs[i].loc['Ionization heights [m]'].dropna()
        bins_arr = np.linspace(z_min, z_max, nbins)
        bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean = utils.get_uncertainties_general(z_ions, 100, bins_arr, Ne, atm_type, z_min, z_max, get_n_H2, str(energies[i]), make_plot = False, make_subset_plot = False, return_kde_curve_median_and_sd = False, plot_each_kde = False, npoints = 500)
        N_peak = utils.get_column_density(peak_loc_mean, z_grid, N_grid)/mH2
        N_ions = utils.get_column_density(z_ions, z_grid, N_grid)/mH2
        bins_arr_N = np.logspace(np.log10(NoverN0_min), np.log10(NoverN0_max), nbins)
        bins_arr_z = utils.get_z_from_column_density(bins_arr_N, z_grid, N_grid)[::-1] # convert R bins to z for normalizing by binwidth
        counts, bins = np.histogram(N_ions/N_peak, bins = bins_arr_N)
        counts = counts / Ne
        binwidths_z = bins_arr_z[1:]-bins_arr_z[:-1]
        q = counts/binwidths_z[::-1]  # this is raw counts / Ntot / binwidth [m]
        plt.step(q, bins_arr_N[1:], color = colors[i], label = str(energies[i]) + ' keV', where = 'post')
    
    ax1 = plt.gca()
    ax1.invert_yaxis()
    ax1.set_xscale('log')
    ax1.set_ylabel(r'Reduced H$_2$ column density', fontsize = fs)
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.legend(fontsize = 0.8*fs)
    plt.semilogy()
    xlims = ax1.get_xlim()
    ylims = ax1.get_ylim()
    plt.ylim([40, 6e-6])
    plt.hlines([1], xmin = xlims[0], xmax = xlims[1], alpha = 0.5, ls = 'dashed', color = 'k')
    plt.savefig(args.out_filepath + '/Jupiter_Galileo_NoverN0_v4.pdf', format = 'pdf', bbox_inches="tight")


## Make Figure 5
if args.fig5:
    
    energies = np.array([10.0])
    df_10 = get_files(args.filepath_Jupiter_galileo, energies)[0]
 
    int_types = ['Ionization heights [m]', 'Elastic scattering heights [m]', 'Vibrational excitation heights [m]', 'Rotational excitation heights [m]', 'B excitation heights [m]', 'C excitation heights [m]', 'a excitation heights [m]', 'b excitation heights [m]', 'c excitation heights [m]', 'e excitation heights [m]', 'Exit (energy) heights [m]']
    labels = ['Ionization', 'Elastic scattering', 'Vibrational excitation', 'Rotational excitation', 'B excitation', 'C excitation', 'a excitation', 'b excitation', 'c excitation', 'e excitation', 'Thermalization']
    plt.figure(figsize = [8,6], dpi = 200)
    fs = 16
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    z_min = 150e3
    z_max = 2200e3
    nbins = 60
    Ne = 1000
    picaso_filepath = './Picaso_data'

    atm_type = 'Jupiter'
    get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)
    alpha = 0.65
    E = 10 # keV

    colors = ['royalblue', 'purple', 'goldenrod', 'goldenrod', 'tomato', 'tomato', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen', 'k']
    alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    markers = ['.', '.', '.', 's', '.', 's', '.', 's', 'd', '^','.']
    switch = [1,2,1,2,1,2,1,2,2,2,1]
    ms = np.array([5, 5, 5, 3, 5, 3, 5, 3, 3, 3, 5])
    lw = 0.65
    for i in range(len(int_types)):
        int_type = int_types[i]
        if int_type != 'Elastic scattering heights [m]':
            z_events = df_10.loc[int_type].dropna()
            bins_arr = np.linspace(z_min, z_max, nbins)
            binwidth = bins_arr[1] - bins_arr[0]
            counts, bins = np.histogram(z_events, bins = bins_arr)
            counts = counts / Ne
            binwidth = (bins[1]-bins[0])
            bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth  
            q = counts/binwidth
            plt.step(q, bins[:-1]/1000, where = 'post', color = colors[i], alpha = alphas[i], lw = lw)
            marker_idxs = (np.abs((np.arange(0,len(bincenters)) % 2 == 0) - (switch[i] % 2))).astype(bool)
            plt.scatter(q[marker_idxs], bincenters[marker_idxs]/1000, color = colors[i], alpha = alphas[i], marker = markers[i], s = ms[i]**2) 
            plt.plot([],[],color = colors[i], alpha = alphas[i], marker = markers[i], markersize = ms[i], label = labels[i], lw = 1)

    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_P_H2(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.set_xlabel(r'Event rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.legend(fontsize = 0.8*fs)
    ax1.set_xlim([1e-6, 4e-2])
    ax1.set_ylim([250,1900])

    plt.savefig(args.out_filepath + '/compare_events_Jupiter_v4.pdf', format = 'pdf', bbox_inches="tight")    


## Make Figure 6
if args.fig6:
    energies =  np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    dfs_Jupiter = get_files(args.filepath_Jupiter_galileo, energies)
    dfs_900K_g5 = get_files(args.filepath_T900_g5, energies)
    
    plt.figure(figsize= [16,6], dpi = 200)
    fs = 16
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    z_min_Jupiter = 150e3
    z_max_Jupiter = 2200e3
    z_min_900K_g5 = 10e3
    z_max_900K_g5 = 30e3
    P_min = 5e-13
    P_max = 1e-3
    N_min = 6e17
    N_max = 1e25
    nbins = 55
    Ne = 1000
    mH2 = 3.347649043E-27 # H2 mass [kg]
    picaso_filepath = './Picaso_data'

    get_n_H2_Jupiter, get_P_H2_Jupiter = utils.construct_profile_Jupiter('Jupiter', z_max_Jupiter, picaso_filepath)
    get_n_H2_900K_g5, get_P_H2_900K_g5 = utils.construct_profiles('T900_g5.0', z_max_900K_g5, picaso_filepath)
    z_grid_Jupiter = np.linspace(z_min_Jupiter, z_max_Jupiter, 1000)
    P_grid_Jupiter = get_P_H2_Jupiter(z_grid_Jupiter) / 1e5
    z_of_P_spline_Jupiter = spline(P_grid_Jupiter[::-1], z_grid_Jupiter[::-1], k=1)
    z_grid_900K_g5 = np.linspace(z_min_900K_g5, z_max_900K_g5, 1000)
    P_grid_900K_g5 = get_P_H2_900K_g5(z_grid_900K_g5) / 1e5
    z_of_P_spline_900K_g5 = spline(P_grid_900K_g5[::-1], z_grid_900K_g5[::-1], k=1)

    plt.subplot(121)
    for i in range(len(energies)):
        if energies[i] in [1, 10, 100]:
            z_ions_Jup = dfs_Jupiter[i].loc['Ionization heights [m]'].dropna()
            P_ions = get_P_H2_Jupiter(z_ions_Jup) / 1e5 # then convert to bars
            bins_arr_P = np.logspace(np.log10(P_min), np.log10(P_max), nbins)
            counts, bins = np.histogram(P_ions, bins = bins_arr_P)
            counts = counts / Ne
            bins_arr_z = z_of_P_spline_Jupiter(bins_arr_P)[::-1] # convert P bins to z for normalizing by binwidth
            binwidths_z = bins_arr_z[1:]-bins_arr_z[:-1]
            q = counts/binwidths_z
            plt.step(q, bins_arr_P[1:], color = colors[i], label = str(energies[i]) + ' keV')  
            z_ions_900K_g5 = dfs_900K_g5[i].loc['Ionization heights [m]'].dropna()
            P_ions = get_P_H2_900K_g5(z_ions_900K_g5) / 1e5 # then convert to bars
            bins_arr_P = np.logspace(np.log10(P_min), np.log10(P_max), nbins)
            counts, bins = np.histogram(P_ions, bins = bins_arr_P)
            counts = counts / Ne
            bins_arr_z = z_of_P_spline_900K_g5(bins_arr_P)[::-1]
            binwidths_z = bins_arr_z[1:]-bins_arr_z[:-1]
            q = counts/binwidths_z
            plt.step(q, bins_arr_P[1:], color = colors[i], ls = (0, (3, 1, 1, 1)))
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.semilogy()
    ax1.xaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax1.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
    ax1.minorticks_on()
    plt.plot([],[], ls = 'solid', color = 'k', label = 'Jupiter: T=125K, log(g)=3.4')
    plt.plot([],[], ls =  (0, (3, 1, 1, 1)), color = 'k', label = 'T-Dwarf: T=900K, log(g)=5.0')
    ax1.legend(fontsize = 0.8*fs) 

    z_grid_Jupiter = np.linspace(0,z_max_Jupiter,100)
    N_grid_Jupiter = utils.construct_R_grid(z_grid_Jupiter, z_min_Jupiter, z_max_Jupiter, get_n_H2_Jupiter)
    z_grid_900K_g5 = np.linspace(0,z_max_900K_g5,100)
    N_grid_900K_g5 = utils.construct_R_grid(z_grid_900K_g5, z_min_900K_g5, z_max_900K_g5, get_n_H2_900K_g5)

    plt.subplot(122)
    for i in range(len(energies)):
        if energies[i] in [1, 10, 100]:
            z_ions_Jup = dfs_Jupiter[i].loc['Ionization heights [m]'].dropna()
            N_ions = utils.get_column_density(z_ions_Jup, z_grid_Jupiter, N_grid_Jupiter)/mH2
            bins_arr_N = np.logspace(np.log10(N_min), np.log10(N_max), nbins)
            bins_arr_z = utils.get_z_from_column_density(bins_arr_N*mH2, z_grid_Jupiter, N_grid_Jupiter)[::-1] # convert R bins to z for normalizing by binwidth
            counts, bins = np.histogram(N_ions, bins = bins_arr_N)
            counts = counts / Ne
            binwidths_z = bins_arr_z[1:]-bins_arr_z[:-1]
            q = counts/binwidths_z[::-1]  # this is raw counts / Ntot / binwidth [m]
            plt.step(q, bins_arr_N[1:], color = colors[i], label = str(energies[i]) + ' keV')
            z_ions_900K_g5 = dfs_900K_g5[i].loc['Ionization heights [m]'].dropna()
            N_ions = utils.get_column_density(z_ions_900K_g5, z_grid_900K_g5, N_grid_900K_g5)/mH2
            bins_arr_N = np.logspace(np.log10(N_min), np.log10(N_max), nbins)
            bins_arr_z = utils.get_z_from_column_density(bins_arr_N*mH2, z_grid_900K_g5, N_grid_900K_g5)[::-1] # convert R bins to z for normalizing by binwidth
            counts, bins = np.histogram(N_ions, bins = bins_arr_N)
            counts = counts / Ne
            binwidths_z = bins_arr_z[1:]-bins_arr_z[:-1]
            q = counts/binwidths_z[::-1]  # this is raw counts / Ntot / binwidth [m]
            plt.step(q, bins_arr_N[1:], color = colors[i], ls = (0, (3, 1, 1, 1)))

    plt.semilogy()
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel(r'H$_2$ column density [m$^{-2}$]', fontsize = fs)
    ax1.invert_yaxis()
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    plt.plot([],[], ls = 'solid', color = 'k', label = 'Jupiter: T=125K, log(g)=3.4')
    plt.plot([],[], ls = (0, (3, 1, 1, 1)), color = 'k', label = 'T-Dwarf: T=900K, log(g)=5.0')

    plt.savefig(args.out_filepath + '/Jupiter_Tdwarf_comparison_v4.pdf', format = 'pdf', bbox_inches="tight")

    
## Make Figure 7 
if args.fig7:
    
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    dfs_Jup = get_files(args.filepath_Jupiter_galileo, energies)
    dfs_900K_g5 = get_files(args.filepath_T900_g5, energies)
    dfs_900K_g4 = get_files(args.filepath_T900_g4, energies)
    dfs_1400K_g5 = get_files(args.filepath_T1400_g5, energies)
    dfs_1400K_g4 = get_files(args.filepath_T1400_g4, energies)
    dfs_2000K_g5 = get_files(args.filepath_T2000_g5, energies)
    dfs_T482_g4point7 = get_files(args.filepath_T482_g4point7, energies)
    
    # collect peak locations and undertainties
    Nsigma = 3#1
    picaso_filepath = './Picaso_data'
    stop_height_pcntle = 0.99 # not used, just to get function to work
    energies = [500.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1]
    Ntot = 1000
    n_energies = len(energies)
    dfs_list = [dfs_Jup, dfs_1400K_g4, dfs_900K_g4, dfs_T482_g4point7, dfs_2000K_g5, dfs_1400K_g5, dfs_900K_g5]
    labels = ['Jupiter', 'T=1400K, log(g)=4.0', 'T=900K, log(g)=4.0', 'T=482K, log(g)=4.7', 'T=2000K, log(g)=5.0', 'T=1400K, log(g)=5.0', 'T=900K, log(g)=5.0']
    atm_types = ['Jupiter', 'T1400_g4.0', 'T900_g4.0', 'T482_g4.7', 'T2000_g5.0', 'T1400_g5.0', 'T900_g5.0']   
    n_atm_types = len(atm_types)
    z_peak_err_arr_lists = []
    z_peak_mean_arr_lists = []
    P_peak_err_arr_lists = []
    P_peak_mean_arr_lists = []        
    N_peak_err_arr_lists = []
    N_peak_mean_arr_lists = []

    for i in range(n_atm_types):
        atm_type = atm_types[i]
        print()
        print(atm_type, ':')

        if atm_type == 'Jupiter':
            z_max = 2000e3
            z_min = 100e3
            RoverR0_min = 1e-6
            RoverR0_max = 5e0
            get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)
            g = 24.79 # m/s^2
        elif atm_type == 'T1400_g4.0':
            z_max = 755.560e3
            z_min = 250e3
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = 250e3
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        elif atm_type == 'T900_g5.0':
            z_max = 37.612e3
            z_min = 10e3
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = 10e3
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        elif atm_type == 'T1400_g5.0':
            z_max = 71.970e3
            z_min = 15e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = z_min
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        elif atm_type == 'T900_g4.0':
            z_max = 398.090e3
            z_min = 140e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = z_min
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        elif atm_type == 'T482_g4.7':
            z_max = 38.942e3 * 1.2
            z_min =  10e3 #???
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = z_min
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        elif atm_type == 'T2000_g5.0':
            z_max = 92.236e3
            z_min =  25e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
            plot_zmin = z_min
            RoverR0_min = 1e-6
            RoverR0_max = 1e0
        else:
             raise ValueError('atm_type must be one of implemented profiles.')
        nbins = 80
        bins_arr = np.linspace(z_min, z_max, nbins)
        n_energies = len(energies)
        z_peak_err_arr = np.zeros([n_energies])*np.nan
        z_peak_mean_arr = np.zeros([n_energies])*np.nan
        P_peak_err_arr = np.zeros([n_energies])*np.nan
        P_peak_mean_arr = np.zeros([n_energies])*np.nan        
        N_peak_err_arr = np.zeros([n_energies])*np.nan
        N_peak_mean_arr = np.zeros([n_energies])*np.nan
        for k in range(len(energies)):
            print(' ' + str(energies[k]) + ' keV')
            z_ions = dfs_list[i][k].loc['Ionization heights [m]'].dropna()
            nfolds = 100
            bins_sd, z_peak_sd, z_peak_mean, P_peak_sd, P_peak_mean, RoverR0_peak_sd, RoverR0_peak_mean, R_peak_sd, R_peak_mean, R0_sd, R0_mean, stopping_P_sd, stopping_P_mean = utils.get_uncertainties(z_ions, nfolds, bins_arr, Ntot, stop_height_pcntle, get_n_H2, get_P_H2, z_max, atm_type, z_min, z_max)
            z_peak_err = z_peak_sd*Nsigma # use N sigma error bars
            P_peak_err = P_peak_sd*Nsigma # use N sigma error bars
            R_peak_err = R_peak_sd*Nsigma # use N sigma error bars
            z_peak_err_arr[k] = z_peak_err
            z_peak_mean_arr[k] = z_peak_mean
            P_peak_err_arr[k] = P_peak_err
            P_peak_mean_arr[k] = P_peak_mean
            N_peak_err_arr[k] = R_peak_err/mH2
            N_peak_mean_arr[k] = R_peak_mean/mH2

        z_peak_err_arr_lists += [z_peak_err_arr]
        z_peak_mean_arr_lists += [z_peak_mean_arr]
        P_peak_err_arr_lists += [P_peak_err_arr]
        P_peak_mean_arr_lists += [P_peak_mean_arr]    
        N_peak_err_arr_lists += [N_peak_err_arr]
        N_peak_mean_arr_lists += [N_peak_mean_arr]   
        
    # make plots -- really no need for this second loop since reading in data in same script now
    size = [18,6]
    fs = 16
    ms = 10
    cs = 0#6
    Nsigma = 3#1
    fig = plt.figure(figsize = size, dpi = 300) # peak height vs e0
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    g = [3.4, 4.0, 4.0, 4.7, 5.0, 5.0, 5.0]
    atm_types = ['Jupiter', 'T1400_g4.0', 'T900_g4.0', 'T482_g4.7', 'T2000_g5.0', 'T1400_g5.0', 'T900_g5.0']
    n_atm_types = len(atm_types)
    energies = [500.0, 100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1]
    Ntot = 1000
    n_energies = len(energies)
    labels = ['Jupiter', 'T=1400K, log(g)=4.0', 'T=900K, log(g)=4.0', 'T=482K, log(g)=4.7', 'T=2000K, log(g)=5.0', 'T=1400K, log(g)=5.0', 'T=900K, log(g)=5.0']
    colors = ['C0', 'C1', 'C1', 'C2', 'C4', 'C4', 'C4']
    markers = ['.', '.', 'D',  '.', '.', 'D', 'v']#, 'h']
    markersize = [ms, ms, 0.5*ms, ms, ms, 0.5*ms,0.6*ms]

    picaso_filepath = './Picaso_data'
    for i in range(n_atm_types):
        atm_type = atm_types[i]
        if atm_type == 'Jupiter':
            z_max = 2000e3
            z_min = 100e3
            get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T1400_g4.0':
            z_max = 755.560e3
            z_min = 250e3
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T900_g5.0':
            z_max = 37.612e3
            z_min = 10e3
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T1400_g5.0':
            z_max = 71.970e3
            z_min = 15e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T900_g4.0':
            z_max = 398.090e3
            z_min = 140e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T482_g4.7':
            z_max = 38.942e3 * 1.2
            z_min =  10e3 #???
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        elif atm_type == 'T2000_g5.0':
            z_max = 92.236e3
            z_min =  25e3 
            get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
        else:
             raise ValueError('atm_type must be one of implemented profiles.')
        z_peak_mean_arr = z_peak_mean_arr_lists[i]
        z_peak_err_arr = z_peak_err_arr_lists[i]
        P_peak_mean_arr = P_peak_mean_arr_lists[i]
        P_peak_err_arr = P_peak_err_arr_lists[i]
        N_peak_mean_arr = N_peak_mean_arr_lists[i]
        N_peak_err_arr = N_peak_err_arr_lists[i]
        # plot peak location height vs e0 for this body
        ax1.errorbar(energies, z_peak_mean_arr/1000, yerr = z_peak_err_arr/1000, fmt = '.', linewidth = 1, elinewidth = 1, capsize = cs*1.15, markersize = markersize[i]*1.15, label = labels[i], color = colors[i], marker = markers[i])#+ r' peak location, '+str(Nsigma)+r'$\sigma$ errorbars')     

        # plot peak location pressure/g vs e0 for this body
        z_grid = np.linspace(0,z_max,10000)
        R_grid = utils.construct_R_grid(z_grid, z_min, z_max, get_n_H2)
        N_peak = utils.get_column_density(z_peak_mean_arr, z_grid, R_grid)/mH2
        N_err = utils.get_column_density(z_peak_mean_arr, z_grid, R_grid)/mH2 - utils.get_column_density(z_peak_mean_arr+z_peak_err_arr, z_grid, R_grid)/mH2 
        ax2.errorbar(energies, N_peak, yerr = N_err, fmt = '.', linewidth = 1, elinewidth = 1, capsize = cs, markersize = markersize[i], label = labels[i], color = colors[i], marker = markers[i])

    ax1.set_xlabel(r'$\varepsilon_0$ [keV]', fontsize = fs)
    ax1.set_ylabel('Peak height [km]', fontsize = fs)
    ax1.semilogx()
    ax1.semilogy()  
    ax1.minorticks_on()

    ax2.set_xlabel(r'$\varepsilon_0$ [keV]', fontsize = fs)
    ax2.set_ylabel(r'Peak column density [m$^{-2}$]', fontsize = fs)
    ax2.semilogx()
    ax2.semilogy()  
    ax2.legend(framealpha = 1, fontsize = 0.8*fs)  
    ax2.minorticks_on()

    plt.savefig(args.out_filepath + '/peak_z_and_N_v5.pdf', format = 'pdf', bbox_inches="tight")


## Make Figure 8
if args.fig8:
    
    # -------- From Updated_pymc_fitting.py (run_updated_pymc_fitting.sh) ----------- #
    energies_T900_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T900_g5 = np.array([2090.0, 9579.0, 17676.0, 85720.0, 158596.0, 746475.0, 1442069.0, 6908740.0, ])
    T900_g5_moyal_mus = np.array([-45.923381, -47.546065, -48.468405, -50.916936, -52.029636, -54.803515, -56.002757, -58.976731])
    T900_g5_moyal_sigmas = np.array([0.564272, 0.489527, 0.467927, 0.46023, 0.443881, 0.450962, 0.44697, 0.44631])
    T900_g5_moyal_mus_sd = np.array([0.019243, 0.007945, 0.005408, 0.002448, 0.001711, 0.000811, 0.000583, 0.000259])
    T900_g5_moyal_sigmas_sd = np.array([0.009823, 0.003972, 0.002858, 0.001238, 0.000905, 0.000417, 0.000305, 0.000136])

    energies_T1400_g4 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T1400_g4 = np.array([2109.0, 9648.0, 18753.0, 85901.0, 158361.0, 773131.0, 1487234.0, 6244032.0, ])
    T1400_g4_moyal_mus = np.array([-45.949328, -47.567456, -48.423349, -50.889231, -52.038044, -54.840218, -56.025174, -58.943613])
    T1400_g4_moyal_sigmas = np.array([0.540444, 0.460409, 0.462888, 0.447978, 0.46457, 0.438082, 0.455741, 0.459023])
    T1400_g4_moyal_mus_sd = np.array([0.018042, 0.007242, 0.0051, 0.002362, 0.001803, 0.000768, 0.000576, 0.000282])
    T1400_g4_moyal_sigmas_sd = np.array([0.009669, 0.003878, 0.002723, 0.001241, 0.000919, 0.000404, 0.000299, 0.000147])

    energies_Jupiter_galileo = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_Jupiter_galileo = np.array([2091.0, 9990.0, 18399.0, 81109.0, 153328.0, 740320.0, 1399427.0, 6115316.0, ])
    Jupiter_galileo_moyal_mus = np.array([-45.927155, -47.569843, -48.4211, -50.868311, -52.076396, -54.782534, -56.00921, -58.854944])
    Jupiter_galileo_moyal_sigmas = np.array([0.518115, 0.47882, 0.480645, 0.454596, 0.431795, 0.453536, 0.45433, 0.443039])
    Jupiter_galileo_moyal_mus_sd = np.array([0.017812, 0.007505, 0.005585, 0.002405, 0.00168, 0.000819, 0.000594, 0.000269])
    Jupiter_galileo_moyal_sigmas_sd = np.array([0.009106, 0.003845, 0.002847, 0.001276, 0.000895, 0.000429, 0.000317, 0.000147])

    energies_T482_g4point7 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T482_g4point7 = np.array([2063.0, 9977.0, 18460.0, 83607.0, 150315.0, 755756.0, 1448214.0, 6599894.0, ])
    T482_g4point7_moyal_mus = np.array([-45.861091, -47.561038, -48.428807, -50.884326, -52.037326, -54.816666, -56.067652, -58.942142])
    T482_g4point7_moyal_sigmas = np.array([0.54052, 0.484312, 0.464949, 0.468281, 0.448523, 0.442007, 0.443687, 0.443699])
    T482_g4point7_moyal_mus_sd = np.array([0.018209, 0.007706, 0.005222, 0.002508, 0.001771, 0.000765, 0.000551, 0.000267])
    T482_g4point7_moyal_sigmas_sd = np.array([0.009455, 0.004063, 0.002852, 0.001329, 0.000954, 0.000424, 0.0003, 0.000145])

    energies_T2000_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T2000_g5 = np.array([2083.0, 9467.0, 18574.0, 81011.0, 153435.0, 787426.0, 1389346.0, 7071487.0, ])
    T2000_g5_moyal_mus = np.array([-45.880346, -47.528091, -48.442424, -50.898345, -52.043203, -54.863353, -56.042302, -58.946484])
    T2000_g5_moyal_sigmas = np.array([0.559586, 0.48293, 0.465221, 0.44169, 0.467042, 0.437353, 0.438405, 0.442334])
    T2000_g5_moyal_mus_sd = np.array([0.01935, 0.007833, 0.005173, 0.002412, 0.001804, 0.000745, 0.000573, 0.000254])
    T2000_g5_moyal_sigmas_sd = np.array([0.010047, 0.004146, 0.002832, 0.001295, 0.000957, 0.000396, 0.000297, 0.000137])

    energies_T900_g4 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T900_g4 = np.array([2027.0, 9758.0, 18327.0, 83691.0, 151719.0, 741253.0, 1411780.0, 7021595.0, ])
    T900_g4_moyal_mus = np.array([-45.884418, -47.526329, -48.444349, -50.895638, -52.033265, -54.836297, -56.059946, -58.95474]) #-61.551228]) <-- It looks like there was accidently several runs in this directory and it fit an old one too?
    T900_g4_moyal_sigmas = np.array([0.575805, 0.490752, 0.473922, 0.441437, 0.456795, 0.445194, 0.436987, 0.436813]) # 0.443358])
    T900_g4_moyal_mus_sd = np.array([0.020137, 0.007891, 0.005334, 0.002405, 0.001826, 0.000818, 0.000557, 0.000256]) # .523557])
    T900_g4_moyal_sigmas_sd = np.array([0.010156, 0.004087, 0.002842, 0.001276, 0.000954, 0.00042, 0.000299, 0.000133]) #0.012798])

    energies_T1400_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T1400_g5 = np.array([2062.0, 9609.0, 18267.0, 78314.0, 158263.0, 726899.0, 1475380.0, 7002785.0, ])
    T1400_g5_moyal_mus = np.array([-45.899517, -47.511262, -48.478905, -50.874445, -52.02968, -54.831663, -56.077233, -58.962087])
    T1400_g5_moyal_sigmas = np.array([0.555463, 0.489775, 0.462846, 0.44241, 0.454407, 0.456712, 0.436711, 0.446569])
    T1400_g5_moyal_mus_sd = np.array([0.018528, 0.007632, 0.005338, 0.002446, 0.001774,  0.00083, 0.000557, 0.000259])
    T1400_g5_moyal_sigmas_sd = np.array([0.009499, 0.004095, 0.002718, 0.001268, 0.000901, 0.000432, 0.000293, 0.000137])
    # ------------------------------------------------------------------------------- #

    atm_types = ['T=900K, log(g)=5.0', 'T=1400K, log(g)=5.0', 'T=2000K, log(g)=5.0', 'T=482K, log(g)=4.7',  'T=900K, log(g)=4.0', 'T=1400K, log(g)=4.0', 'Jupiter']
    mus_list = [T900_g5_moyal_mus,  T1400_g5_moyal_mus,  T2000_g5_moyal_mus,  T482_g4point7_moyal_mus,  T900_g4_moyal_mus, T1400_g4_moyal_mus, Jupiter_galileo_moyal_mus]
    sigmas_list =  [T900_g5_moyal_sigmas,  T1400_g5_moyal_sigmas,  T2000_g5_moyal_sigmas,  T482_g4point7_moyal_sigmas,  T900_g4_moyal_sigmas, T1400_g4_moyal_sigmas, Jupiter_galileo_moyal_sigmas]
    mus_sd_list =  [T900_g5_moyal_mus_sd,  T1400_g5_moyal_mus_sd,  T2000_g5_moyal_mus_sd,  T482_g4point7_moyal_mus_sd,  T900_g4_moyal_mus_sd, T1400_g4_moyal_mus_sd, Jupiter_galileo_moyal_mus_sd]
    sigmas_sd_list =  [T900_g5_moyal_sigmas_sd,  T1400_g5_moyal_sigmas_sd,  T2000_g5_moyal_sigmas_sd,  T482_g4point7_moyal_sigmas_sd,  T900_g4_moyal_sigmas_sd, T1400_g4_moyal_sigmas_sd, Jupiter_galileo_moyal_sigmas_sd]
    energies_list =  [energies_T900_g5,  energies_T1400_g5,  energies_T2000_g5,  energies_T482_g4point7,  energies_T900_g4, energies_T1400_g4, energies_Jupiter_galileo]
       
    n=8
    fs = 16
    cmap = mcp.gen_color(cmap="plasma",n=n)
    lws = [7,6,5,4,3,2,1]
    E_arr = np.logspace(-1,2.7,100)    
    plt.figure(figsize = [18,6], dpi = 300)
    plt.subplot(121)
    ax = plt.gca()
    colors = ['C4', 'C4', 'C4', 'C2', 'C1', 'C1', 'C0']
    markers = ['v', 'D','.', '.','D','.', '.',]
    ms = 10
    markersize = [0.6*ms, 0.5*ms, ms, ms,0.5*ms,ms, ms]
    for i in range(len(mus_list))[::-1]:
        p=plt.errorbar(energies_list[i], mus_list[i], yerr=mus_sd_list[i], label = atm_types[i], color = colors[i], fmt = markers[i], ms = markersize[i], alpha = 0.8)
    plt.plot(E_arr, utils.moyal_mu(E_arr), label = 'Polynomial fit', color = 'k', ls = 'dashed', alpha = 0.6)
    plt.xlabel(r'$\varepsilon_0$ [keV]',fontsize=fs)
    plt.ylabel(r'$\mu$',fontsize=fs)
    plt.semilogx()
    plt.xlim([7e-2, 6e2])
    handles, labels = plt.gca().get_legend_handles_labels() 
    order = [1,2,3,4,5,6,0] 
    plt.legend([handles[i] for i in order], [labels[i] for i in order],fontsize=fs*0.8)

    plt.subplot(122)
    ax = plt.gca()
    for i in range(len(mus_list))[::-1]:
        label = atm_types[i]
        p=plt.errorbar(energies_list[i], sigmas_list[i], yerr=sigmas_sd_list[i], label = label, color = colors[i], fmt = markers[i], ms = markersize[i], alpha = 0.8)
    plt.plot(E_arr, utils.moyal_sigma(E_arr), color = 'k', ls = 'dashed', alpha = 0.6)
    plt.xlabel(r'$\varepsilon_0$ [keV]',fontsize=fs)
    plt.ylabel(r'$\sigma$',fontsize=fs)
    plt.semilogx()

    plt.savefig(args.out_filepath + '/mu_and_sigma_transformed_v4.pdf', format = 'pdf', bbox_inches="tight")


## Make Figure 9
if args.fig9:   
    
    Ne = 1000
    
    # -------- Copy Nions from output files from Updated_pymc_fitting.py (run_updated_pymc_fitting.sh) ----------- #
    Nionizations_T900_g5 = np.array([2090.0, 9579.0, 17676.0, 85720.0, 158596.0, 746475.0, 1442069.0, 6908740.0, ])
    energies_T900_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T1400_g4 = np.array([2109.0, 9648.0, 18753.0, 85901.0, 158361.0, 773131.0, 1487234.0, 6244032.0, ])
    energies_T1400_g4 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_Jupiter = np.array([2091.0, 9990.0, 18399.0, 81109.0, 153328.0, 740320.0, 1399427.0, 6115316.0, ])
    energies_Jupiter = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T482_g4point7 = np.array([2063.0, 9977.0, 18460.0, 83607.0, 150315.0, 755756.0, 1448214.0, 6599894.0, ])
    energies_T482_g4point7 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T2000_g5 = np.array([2083.0, 9467.0, 18574.0, 81011.0, 153435.0, 787426.0, 1389346.0, 7071487.0, ])
    energies_T2000_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T900_g4 = np.array([2027.0, 9758.0, 18327.0, 83691.0, 151719.0, 741253.0, 1411780.0, 7021595.0, ])
    energies_T900_g4 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    Nionizations_T1400_g5 = np.array([2062.0, 9609.0, 18267.0, 78314.0, 158263.0, 726899.0, 1475380.0, 7002785.0, ])
    energies_T1400_g5 = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, ])
    # ------------------------------------------------------------------------------------------------------------- #

    atm_types = ['T=900K, log(g)=5.0', 'T=1400K, log(g)=5.0', 'T=2000K, log(g)=5.0', 'T=482K, log(g)=4.7',  'T=900K, log(g)=4.0', 'T=1400K, log(g)=4.0', 'Jupiter']
    energies_list =  [energies_T900_g5,  energies_T1400_g5,  energies_T2000_g5,  energies_T482_g4point7,  energies_T900_g4, energies_T1400_g4, energies_Jupiter]  
    Nions = [Nionizations_T900_g5, Nionizations_T1400_g5, Nionizations_T2000_g5, Nionizations_T482_g4point7, Nionizations_T900_g4, Nionizations_T1400_g4, Nionizations_Jupiter]
    Nions_flat = [x/Ne for xs in Nions for x in xs]
    energies = np.array([0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0])
    atm_types = ['T=900K, log(g)=5.0', 'T=1400K, log(g)=5.0','T=2000K, log(g)=5.0','T=482K, log(g)=4.7','T=900K, log(g)=4.0','T=1400K, log(g)=4.0','Jupiter'] # isothermal jupiter
    colors = ['C4', 'C4', 'C4', 'C2', 'C1', 'C1', 'C0']
    markers = ['v', 'D','.', '.','D','.', '.',]#, 'h']
    ms = 10
    markersize = [0.6*ms, 0.5*ms, ms, ms,0.5*ms,ms, ms]

    plt.figure(figsize = [8,6], dpi = 200)
    fs = 16
    for i in range(len(Nions))[::-1]:
        plt.errorbar(energies, np.array(Nions[i])/Ne, yerr = np.array(np.sqrt(Nions[i]))/Ne, color = colors[i], fmt = markers[i], markersize = markersize[i], label = atm_types[i])
        plt.xlabel(r'$\varepsilon_0$ [keV]', fontsize = fs)
        plt.ylabel(r'$\beta_{Ionization}$', fontsize = fs)

    E_arr = np.logspace(-1,2.7,100)
    plt.plot(E_arr, utils.calc_Nevent_over_Ne(E_arr, 'Ionization heights [m]'), color = 'k', ls = 'dashed', label = 'Fit', alpha = 0.6)
    plt.semilogx()
    plt.semilogy()
    plt.legend(fontsize = 0.8*fs)

    handles, labels = plt.gca().get_legend_handles_labels() 
    order = [1,2,3,4,5,6,7,0] 
    plt.legend([handles[i] for i in order], [labels[i] for i in order],fontsize=fs*0.8)
    plt.savefig(args.out_filepath + '/beta_v2.pdf', format = 'pdf', bbox_inches="tight")


## Make Figure 10
if args.fig10:   
    atm_type = 'T900_g5.0'
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    dfs_900K_g5 = get_files(args.filepath_T900_g5, energies)
    picaso_filepath = './Picaso_data'
    z_max = 37.612e3 * 1.2
    z_min = 0e3
    get_n_H2, get_pressure = utils.construct_profiles(atm_type, z_max, picaso_filepath)
    Ne = 1000
    z = np.linspace(z_min, z_max-1, 1000) # m
    nbins = 100
    bins_arr = np.linspace(z_min, z_max, nbins)
    binwidth = bins_arr[1] - bins_arr[0]
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, z_max, picaso_filepath)
    bins_sd_list = []
    kde_curve_medians = []
    kde_curve_sds = []
    npoints = 500
    
    # collect kde curve values
    for i in range(len(energies)): 
        print(str(energies[i]) + ' keV: ')
        nfolds = 100
        z_ions = dfs_900K_g5[i].loc['Ionization heights [m]'].dropna()
        Ntot = len(z_ions)
        min_val = z_min
        max_val = z_max 
        bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean, kde_curve_median, kde_curve_sd = utils.get_uncertainties_general(z_ions, nfolds, bins_arr, Ntot, atm_type, min_val, max_val, get_n_H2, str(energies[i]) + '_' + atm_type, make_plot = False, make_subset_plot = False, return_kde_curve_median_and_sd = True, plot_each_kde = False, npoints = npoints)
        bins_err = 3*bins_sd/binwidth
        bins_sd_list += [bins_sd]
        kde_curve_medians += [kde_curve_median*Ntot/Ne]
        kde_curve_sds += [kde_curve_sd*Ntot/Ne]

    # make the plots       
    plt.figure(figsize = [8,6], dpi = 300)
    fs = 16
    energies = [500, 100, 50, 10, 5, 1, 0.5, 0.1]
    npoints = len(kde_curve_medians[0])
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    nbins = 80
    Ne = 1000
    kde_z = np.linspace(z_min, z_max, npoints)
    param_z = np.linspace(z_min, z_max, 500)
    picaso_filepath = './Picaso_data'

    alpha = 0.65
    for i in range(len(energies)):
        if energies[i] in [5,50,500]:
            plt.step([], [], color = colors[i], label = str(energies[i]) + ' keV', where = 'post')
            plt.plot(utils.calc_q(param_z, energies[i], z_min, z_max, get_n_H2, 'Ionization heights [m]'), param_z/1e3, ls = 'dashed', alpha = 1.0, color = cmap[i])
            plt.fill_betweenx(kde_z/1e3, kde_curve_medians[i]-3*kde_curve_sds[i], kde_curve_medians[i]+3*kde_curve_sds[i], color = cmap[i], alpha = 0.25)
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_P_H2(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.plot([],[], ls = 'dashed', color = 'k', label = 'Parameterization')
    ax1.fill_betweenx([],[], ls = 'solid', color = 'k', alpha = 0.25, label = 'Simulation results')

    ax1.legend(fontsize = 0.8*fs)
    ax1.set_xlim([1e-5, 1e1])
    ax1.set_ylim([5,32])
    plt.savefig(args.out_filepath + '/900K_g5_q_kde_vs_parameterization_v4.png')#, format = 'pdf', bbox_inches="tight")


## Make Figure 11
if args.fig11:
    
    atm_type = 'T900_g5.0'   
    z_max = 37.612e3
    z_min = 10e3
    Ne = 1000
    
    # For the T=900K, logg=5 Tdwarf, as a function of energy
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    dfs_900K_g5 = get_files(args.filepath_T900_g5, energies)
    
    kde_peak_locs = []
    kde_peak_vals = []
    kde_peak_locs_sd = []
    kde_peak_vals_sd = []
    parameterization_peak_locs = []
    parameterization_peak_vals = []
    npoints = 500
    z = np.linspace(z_min, z_max, 1000) # m 
    nbins = 80
    bins_arr = np.linspace(z_min, z_max, nbins)
    binwidth = bins_arr[1] - bins_arr[0]
    picaso_filepath = './Picaso_data'
    get_n_H2, get_pressure = utils.construct_profiles(atm_type, z_max, picaso_filepath)
    
    # collect values of parameterization loc and value for smoother curve (vs just at simulation energies as used for residuals)
    E_arr = np.logspace(-1,np.log10(500),500)
    parameterization_peak_locs_smooth = np.zeros(len(E_arr))
    parameterization_peak_vals_smooth = np.zeros(len(E_arr))
    for i in range(len(E_arr)):
        q_ion_param = utils.calc_q(z, E_arr[i], z_min, z_max, get_n_H2, 'Ionization heights [m]')
        parameterization_peak_locs_smooth[i] = z[q_ion_param == np.nanmax(q_ion_param)][0]
        parameterization_peak_vals_smooth[i] = np.nanmax(q_ion_param)
    
    # collect kde curve values
    for i in range(len(energies)): 
        print(str(energies[i]) + ' keV: ')
        nfolds = 100
        z_ions = dfs_900K_g5[i].loc['Ionization heights [m]'].dropna()
        Ntot = len(z_ions)
        min_val = z_min
        max_val = z_max 
        bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean, kde_curve_median, kde_curve_sd = utils.get_uncertainties_general(z_ions, nfolds, bins_arr, Ntot, atm_type, min_val, max_val, get_n_H2, str(energies[i]) + '_' + atm_type, make_plot = False, make_subset_plot = False, return_kde_curve_median_and_sd = True, plot_each_kde = False, npoints = npoints)
        bins_err = 3*bins_sd/binwidth
        kde_peak_locs += [peak_loc_mean]
        kde_peak_vals += [peak_val_mean*Ntot/Ne]
        kde_peak_locs_sd += [peak_loc_sd]
        kde_peak_vals_sd += [peak_val_sd*Ntot/Ne]        
        q_ion_param = utils.calc_q(z, energies[i], z_min, z_max, get_n_H2, 'Ionization heights [m]')
             
        parameterization_peak_locs += [z[q_ion_param == np.nanmax(q_ion_param)][0]]
        parameterization_peak_vals += [np.nanmax(q_ion_param)]   
    
    parameterization_peak_locs = np.array(parameterization_peak_locs)
    parameterization_peak_vals = np.array(parameterization_peak_vals)
    
    fs = 14
    fig, ax = plt.subplots(2, 2, gridspec_kw={'width_ratios': [1, 1], 'height_ratios': [1, 0.5]},figsize = [18,6], dpi = 200)
    colors = ['k', 'k']
    symbols = ['.', 's']
    markersizes = [8,3]
    alpha = 1.0
    cmap = mcp.gen_color(cmap="plasma",n=9)

    plt.subplot(221)
    for i in range(len(energies)):
        plt.plot(energies[i], np.array(kde_peak_locs)[i]/1e3, markersize = markersizes[0],  marker = symbols[0], color =  cmap[i], linestyle = 'none')#, label = 'Simulation') 
        plt.errorbar(energies[i], np.array(kde_peak_locs)[i]/1e3, yerr = np.array(kde_peak_locs_sd)[i]/1e3, markersize = markersizes[0],  marker = symbols[0], color =  cmap[i], alpha = alpha, linestyle = 'none')#, label = 'Simulation') 
    plt.plot(E_arr, parameterization_peak_locs_smooth/1e3, ls ='dashed', alpha = 0.7, color = colors[1], label = 'Parameterization')
    plt.semilogx()
    plt.ylabel(r'q$_{ion}$ peak altitude [km]',fontsize = fs)
    plt.errorbar([], [], yerr = [], markersize = markersizes[0],  marker = symbols[0], color = colors[0], linestyle = 'none', label = 'Simulation') 
    plt.legend(fontsize = 0.8*fs)
    plt.gca().minorticks_on()

    plt.subplot(222)
    for i in range(len(energies)):
        plt.plot(energies[i], np.array(kde_peak_vals)[i], markersize = markersizes[0],  marker = symbols[0], color = cmap[i], linestyle = 'none')#, label = 'Simulation') 
        plt.errorbar(energies[i], np.array(kde_peak_vals)[i], yerr = np.array(kde_peak_vals_sd)[i], markersize = markersizes[0], marker = symbols[0], color = cmap[i], alpha = alpha, linestyle = 'none')#, label = 'Simulation') 
    plt.plot(E_arr, parameterization_peak_vals_smooth, ls ='dashed', alpha = 0.7, color = colors[1], label = 'Parameterization')
    plt.semilogx()
    plt.semilogy()
    plt.ylabel(r'q$_{ion}$ peak value [m$^{-1}$]',fontsize = fs)
    plt.errorbar([], [], yerr = [], markersize = markersizes[0],  marker = symbols[0], color = colors[0], linestyle = 'none', label = 'Simulation') 
    plt.legend(fontsize = 0.8*fs)

    plt.subplot(223)
    for i in range(len(energies)):
        plt.plot(energies[i], (np.array(kde_peak_locs)[i] - parameterization_peak_locs[i])/np.array(kde_peak_locs)[i], markersize = markersizes[0],  marker = symbols[0], color =  cmap[i], linestyle = 'none')#, label = 'Simulation') 
    plt.semilogx()
    plt.xlabel(r'$\varepsilon_0$ [keV]',fontsize = fs)
    plt.ylabel('Fractional error',fontsize = fs)

    plt.subplot(224)
    for i in range(len(energies)):
        plt.plot(energies[i], (np.array(kde_peak_vals)[i] - parameterization_peak_vals[i])/np.array(kde_peak_vals)[i], markersize = markersizes[0],  marker = symbols[0], color =  cmap[i], linestyle = 'none')#, label = 'Simulation') 
    plt.semilogx()
    plt.xlabel(r'$\varepsilon_0$ [keV]',fontsize = fs)
    plt.ylabel('Fractional error',fontsize = fs)

    plt.savefig(args.out_filepath + '/900K_g5_peak_vals_and_locs_residuals_v3.pdf', format = 'pdf', bbox_inches="tight")

## Make Figure 12
if args.fig12:
    atm_type = 'Jupiter'
    energies = np.array([500, 100, 50, 10, 5, 1, 0.5, 0.1])
    dfs_900K_g5 = get_files(args.filepath_Jupiter_galileo, energies)
    picaso_filepath = './Picaso_data'
    z_min = 150e3
    z_max = 2200e3
    Ne = 1000
    z = np.linspace(z_min, z_max-1, 1000) # m
    nbins = 100
    bins_arr = np.linspace(z_min, z_max, nbins)
    binwidth = bins_arr[1] - bins_arr[0]
    get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, z_max, picaso_filepath)
    bins_sd_list = []
    kde_curve_medians = []
    kde_curve_sds = []
    npoints = 500
    
    # collect ske curve values
    for i in range(len(energies)): 
        print(str(energies[i]) + ' keV: ')
        nfolds = 100
        z_ions = dfs_900K_g5[i].loc['Ionization heights [m]'].dropna()
        Ntot = len(z_ions)
        min_val = z_min
        max_val = z_max 
        bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean, kde_curve_median, kde_curve_sd = utils.get_uncertainties_general(z_ions, nfolds, bins_arr, Ntot, atm_type, min_val, max_val, get_n_H2, str(energies[i]) + '_' + atm_type, make_plot = False, make_subset_plot = False, return_kde_curve_median_and_sd = True, plot_each_kde = False, npoints = npoints)
        bins_err = 3*bins_sd/binwidth
        bins_sd_list += [bins_sd]
        kde_curve_medians += [kde_curve_median*Ntot/Ne]
        kde_curve_sds += [kde_curve_sd*Ntot/Ne]

    # make the plots       
    plt.figure(figsize = [8,6], dpi = 300)
    fs = 16
    energies = [500, 100, 50, 10, 5, 1, 0.5, 0.1]
    npoints = len(kde_curve_medians[0])
    cmap = mcp.gen_color(cmap="plasma",n=9)
    colors = cmap
    nbins = 80
    Ne = 1000
    kde_z = np.linspace(z_min, z_max, npoints)
    param_z = np.linspace(z_min, z_max, 500)
    picaso_filepath = './Picaso_data'

    alpha = 0.65
    for i in range(len(energies)):
        if energies[i] in [5,50,500]:
            plt.step([], [], color = colors[i], label = str(energies[i]) + ' keV', where = 'post')
            plt.plot(utils.calc_q(param_z, energies[i], z_min, z_max, get_n_H2, 'Ionization heights [m]'), param_z/1e3, ls = 'dashed', alpha = 1.0, color = cmap[i])
            plt.fill_betweenx(kde_z/1e3, kde_curve_medians[i]-3*kde_curve_sds[i], kde_curve_medians[i]+3*kde_curve_sds[i], color = cmap[i], alpha = 0.25)
    ax1 = plt.gca()
    ax1.set_xscale('log')
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_P_H2(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.set_xlabel(r'Ionization rate $q_{ion}$ [m$^{-1}$]', fontsize = fs)
    ax1.minorticks_on()
    ax1.plot([],[], ls = 'dashed', color = 'k', label = 'Parameterization')
    ax1.fill_betweenx([],[], ls = 'solid', color = 'k', alpha = 0.25, label = 'Simulation results')

    ax1.legend(fontsize = 0.8*fs)
    ax1.set_xlim([1e-5, 3e-1])
    ax1.set_ylim([100,1300])
    plt.savefig(args.out_filepath + '/Jupiter_galileo_q_kde_vs_parameterization_v5.png')#, format = 'pdf', bbox_inches="tight")

    
## Make Figure 13
if args.fig13:
    def F_high_energy(E_eV, A = 1e10 * (u.cm**(-2)/u.s), e0 = (100*u.keV).to(u.eV)):
        F =  A*E_eV/e0**2 * np.exp(-E_eV/e0)
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / cm^2 / s / eV

    def F_power_law(E_eV):
        A = 1e9 * (u.cm**(-2)/u.s)
        B = 0.5
        e1 = 100 * u.eV
        e2 = (200 * u.keV).to(u.eV)
        scale = B * e1**B * e2**B/(e2**B - e1**B)
        idxs = (E_eV < e1) | (E_eV > e2)
        F = A*E_eV**(-B) * (scale/E_eV)
        F[idxs] = 0
        F = F.to((u.cm**(-2)/ (u.s * u.eV)))
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / cm^2 / s / eV

    def F_triple_maxwellian(E_eV):
        e1 = (0.1*u.keV).to(u.eV)
        e2 = (3*u.keV).to(u.eV)
        e3 = (22*u.keV).to(u.eV)
        A1 = (0.5 * u.erg/ (u.cm**2 * u.s))/e1
        A2 = (10 * u.erg/ (u.cm**2 * u.s))/e2
        A3 = (100 * u.erg/ (u.cm**2 * u.s))/e3
        F =  (A1*F_high_energy(E_eV, 1, e1) + A2*F_high_energy(E_eV, 1, e2) + A3*F_high_energy(E_eV, 1, e3)).to((u.cm**(-2)/ (u.s * u.eV)))
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / m^2 / s / eV
    atm_type = 'T900_g5.0'
    picaso_filepath = './Picaso_data'
    z_max = 37.612e3 * 1.2
    z_min = 10e3
    get_n_H2, get_pressure = utils.construct_profiles(atm_type, z_max, picaso_filepath)

    atm_type = 'T900_g5.0'

    event_type = 'Ionization heights [m]'
    Q_ion_high_energy = []
    Q_ion_power_law = []
    Q_ion_triple_maxwellian = []
    z_arr = np.linspace(0, z_max, 100)
    E_eV = np.logspace(-1, 6.5, 1000) # eV 
    E_keV = E_eV / 1e3
    for i in range(len(z_arr)):
        zi = z_arr[i]
        Q_ion_high_energy += [utils.calc_Q(F_high_energy, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]
        Q_ion_power_law += [utils.calc_Q(F_power_law, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]
        Q_ion_triple_maxwellian += [utils.calc_Q(F_triple_maxwellian, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]

    fs = 16

    # plot energy spectra
    plt.figure(figsize = [18,6], dpi = 300)
    plt.subplot(121)
    plt.plot(E_eV, F_high_energy(E_eV*u.eV)*1e4, label = 'High energy')
    plt.plot(E_eV, F_triple_maxwellian(E_eV*u.eV)*1e4, label = 'Triple maxwellian')
    plt.plot(E_eV[F_power_law(E_eV*u.eV)>0], F_power_law(E_eV*u.eV)[F_power_law(E_eV*u.eV)>0]*1e4, label = 'Power Law')
    plt.ylabel(r'F($\varepsilon_0$) [electrons/m$^2$/s/eV]', fontsize = fs)
    plt.xlabel(r'$\varepsilon_0$ [eV]', fontsize = fs)
    plt.legend(fontsize = 0.8*fs)
    plt.semilogx()
    plt.semilogy()
    plt.ylim([10*1e4,3e7*1e4])
    ax = plt.gca()
    ax.yaxis.set_major_locator(mticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

    plt.subplot(122)
    plt.plot(Q_ion_high_energy, z_arr/1e3, color = 'C0', ls = 'solid', label = 'High energy')#, sim, results w/ interpolated #ions')
    plt.plot(Q_ion_triple_maxwellian, z_arr/1e3, color = 'C1', ls = 'solid', label = 'Triple maxwellian')#, sim, results w/ interpolated #ions')
    plt.plot(Q_ion_power_law, z_arr/1e3, color = 'C2', ls = 'solid', label = 'Power Law')#, sim, results w/ interpolated #ions')

    plt.ylabel('Altitude [km]', fontsize = fs)
    plt.xlabel(r'Q$_{ion}$ [ionizations/m$^3$/s]', fontsize = fs)
    leg = plt.legend(title = 'T=900K, log(g)=5', fontsize = 0.8*fs, title_fontsize= 0.8*fs)
    leg._legend_box.align = "left"
    plt.semilogx()
    plt.ylim([5,44])
    plt.xlim([1e0*1e6,1e8*1e6])
    ax1 = plt.gca()
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_pressure(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.minorticks_on()

    plt.savefig(args.out_filepath + '/Qion_'+atm_type+'_v6.pdf', format = 'pdf', bbox_inches="tight")

    
## Make Figure 14
if args.fig14:
    def F_high_energy(E_eV, A = 1e10 * (u.cm**(-2)/u.s), e0 = (100*u.keV).to(u.eV)):
        F =  A*E_eV/e0**2 * np.exp(-E_eV/e0)
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / cm^2 / s / eV

    def F_power_law(E_eV):
        A = 1e9 * (u.cm**(-2)/u.s)
        B = 0.5
        e1 = 100 * u.eV
        e2 = (200 * u.keV).to(u.eV)
        scale = B * e1**B * e2**B/(e2**B - e1**B)
        idxs = (E_eV < e1) | (E_eV > e2)
        F = A*E_eV**(-B) * (scale/E_eV)
        F[idxs] = 0
        F = F.to((u.cm**(-2)/ (u.s * u.eV)))
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / cm^2 / s / eV

    def F_triple_maxwellian(E_eV):
        e1 = (0.1*u.keV).to(u.eV)
        e2 = (3*u.keV).to(u.eV)
        e3 = (22*u.keV).to(u.eV)
        A1 = (0.5 * u.erg/ (u.cm**2 * u.s))/e1
        A2 = (10 * u.erg/ (u.cm**2 * u.s))/e2
        A3 = (100 * u.erg/ (u.cm**2 * u.s))/e3
        F =  (A1*F_high_energy(E_eV, 1, e1) + A2*F_high_energy(E_eV, 1, e2) + A3*F_high_energy(E_eV, 1, e3)).to((u.cm**(-2)/ (u.s * u.eV)))
        return F#.to(u.eV**-1 * u.s**-1 * u.m**-2).value # e- / m^2 / s / eV
    atm_type = 'T900_g5.0'
    picaso_filepath = './Picaso_data'
    z_max = 37.612e3 * 1.2
    z_min = 10e3
    get_n_H2, get_pressure = utils.construct_profiles(atm_type, z_max, picaso_filepath)

    atm_type = 'T900_g5.0'

    event_type = 'Total energy deposition [eV]'
    Q_high_energy = []
    Q_power_law = []
    Q_triple_maxwellian = []
    z_arr = np.linspace(0, z_max, 100)
    E_eV = np.logspace(-1, 6.5, 1000) # eV 
    E_keV = E_eV / 1e3
    for i in range(len(z_arr)):
        zi = z_arr[i]
        Q_high_energy += [utils.calc_Q(F_high_energy, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]
        Q_power_law += [utils.calc_Q(F_power_law, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]
        Q_triple_maxwellian += [utils.calc_Q(F_triple_maxwellian, zi, E_eV, z_min, z_max, get_n_H2, event_type).to((u.m**-3/u.s)).value]

    fs = 16

    # plot energy spectra
    plt.figure(figsize = [8,6], dpi = 300)
    
    Q_high_energy_ergs = np.array(Q_high_energy) * 1.602176633e-12
    Q_triple_maxwellian_ergs = np.array(Q_triple_maxwellian) * 1.602176633e-12
    Q_power_law_ergs = np.array(Q_power_law_ergs) * 1.602176633e-12
    
    plt.plot(Q_high_energy_ergs, z_arr/1e3, color = 'C0', ls = 'solid', label = 'High energy')#, sim, results w/ interpolated #ions')
    plt.plot(Q_triple_maxwellian_ergs, z_arr/1e3, color = 'C1', ls = 'solid', label = 'Triple maxwellian')#, sim, results w/ interpolated #ions')
    plt.plot(Q_power_law_ergs, z_arr/1e3, color = 'C2', ls = 'solid', label = 'Power Law')#, sim, results w/ interpolated #ions')

    plt.ylabel('Altitude [km]', fontsize = fs)
    plt.xlabel(r'Energy deposition [ergs/m$^3$/s]', fontsize = fs)
    leg = plt.legend(title = 'T=900K, log(g)=5', fontsize = 0.8*fs, title_fontsize= 0.8*fs)
    leg._legend_box.align = "left"
    plt.semilogx()
    plt.ylim([5,44])
    plt.xlim([1e1*1.602176633e-12*1e6,1e9*1.602176633e-12*1e6])
    ax1 = plt.gca()
    ax1.set_ylabel('Altitude [km]', fontsize = fs)
    ytick_locs = ax1.get_yticks() # returns locs in data coords
    P = get_pressure(ytick_locs*1000) / 1e5 # then convert to bars
    ax2 = ax1.twinx()
    ax2.set_yticks(ytick_locs)
    formatting_function = np.vectorize(lambda f: format(f, '6.2E'))
    formatted_P = formatting_function(P)
    label_P = [P.replace('E',r'$\times 10^{')+'}$' for P in formatted_P]
    ax2.set_yticklabels(label_P)
    ax2.set_ybound(ax1.get_ybound())
    ax2.set_ylabel(r'H$_2$ Partial Pressure [bar]', fontsize = fs)
    ax2.minorticks_on()
    ax1.minorticks_on()

    plt.savefig(args.out_filepath + '/Energy_deposition_'+atm_type+'_v8.pdf', format = 'pdf', bbox_inches="tight")


## Make Figure 15
if args.fig15:
    E_rot_excitation = 0.0438 # [eV] * u.eV # 50 * u.eV # [eV] energy lost to excitation in excitation interaction, from MCCC database "threshold energy"
    E_vib_excitation = 8.77596948e-01 # [eV] * u.eV # "threshold energy" from MCCC database
    E_C_excitation = 1.22910e1 # [eV] * u.eV # "threshold energy" from MCCC database
    E_B_excitation = 1.11829e1 # [eV] * u.eV # "threshold energy" from MCCC database
    E_a_excitation = 1.17934e1 # [eV] * u.eV # "threshold energy" from MCCC database
    E_b_excitation = 4.47713e0 # [eV] * u.eV # "threshold energy" from MCCC database
    E_c_excitation = 1.21569e1 # [eV] * u.eV # "threshold energy" from MCCC database
    E_e_excitation = 1.32260E+01 # [eV] * u.eV # "threshold energy" from MCCC database
    E_threshold = E_rot_excitation #1.60218e-19*6.241509e18 #[eV] # 1 eV in J, energy for which ionization cross section equation doesn't hold
    E_ion = 15.43 # [eV] * u.eV # eV to ionize H2 (or is it twice this?). This is "binding energy" on NIST, is that same thing?
    fs = 16
    E_min = 0.01
    E_max = 1e5
    E = np.logspace(np.log10(E_min),np.log10(E_max),500)  # eV
    E_markers = np.logspace(np.log10(E_min),np.log10(E_max),50)  # eV
    xsec1 = utils.ionization_xsec(E/6.242e+18) # convert enrgy to J
    xsec2 = utils.rot_excitation_xsec(E/6.242e+18)
    xsec3 = utils.elastic_scat_xsec(E/6.242e+18)
    xsec4 = utils.B_excitation_xsec(E/6.242e+18)
    xsec5 = utils.C_excitation_xsec(E/6.242e+18)
    xsec6 = utils.vib_excitation_xsec(E/6.242e+18)
    xsec7 = utils.b_excitation_xsec(E/6.242e+18)
    xsec8 = utils.c_excitation_xsec(E/6.242e+18)
    xsec9 = utils.e_excitation_xsec(E/6.242e+18)
    xsec10 = utils.a_excitation_xsec(E/6.242e+18)
    xsecs = [utils.ionization_xsec, utils.elastic_scat_xsec, utils.rot_excitation_xsec, utils.vib_excitation_xsec, utils.B_excitation_xsec, utils.C_excitation_xsec, utils.a_excitation_xsec, utils.b_excitation_xsec, utils.c_excitation_xsec, utils.e_excitation_xsec]
    labels = ['Ionization', 'Elastic scattering', 'Rotational excitation', 'Vibrational excitation', 'B excitaiton', 'C excitation', 'a excitation', 'b excitation', 'c excitation', 'e excitation' ]
    xsec_tot = xsec1 + xsec2 + xsec3 + xsec4 + xsec5 + xsec6 + xsec7 + xsec8 + xsec9 + xsec10
    xsec_tot[E < E_threshold] = xsec3[E < E_threshold]
    plt.figure(figsize= [16,6], dpi = 200)
    colors = ['royalblue', 'purple', 'goldenrod', 'goldenrod', 'tomato', 'tomato', 'forestgreen', 'forestgreen', 'forestgreen', 'forestgreen']
    markers = ['.', '.', '.', 's', '.', 's', '.', 's', 'd', '^']
    ms = np.array([6, 6, 6, 3, 6, 3, 6, 3, 3, 3, 6]) * 1.2
    alpha = 0.7
    for i in range(len(xsecs)):
        plt.plot(E, xsecs[i](E/6.242e+18), color = colors[i], alpha = 0.5)
        plt.plot(E_markers, xsecs[i](E_markers/6.242e+18), markers[i], color = colors[i], markersize = ms[i])
        plt.plot([],[], markers[i], ls = 'solid', color = colors[i], markersize = ms[i], label = labels[i], alpha = alpha)
        
    plt.plot(E, xsec_tot, label = 'Total', color = 'k', ls = 'solid', alpha = alpha)
    plt.semilogx()
    plt.semilogy()
    ylims = plt.gca().get_ylim()
    a = 0.5
    thresholds_style = 'dashed'
    plt.vlines([E_threshold], ymin = ylims[0], ymax = ylims[1], alpha = a, ls = thresholds_style, color = 'k', label = 'Thermalization\nthreshold')
    ax = plt.gca()
    arrowmin = 2e-24
    arrowmax = 5e-24
    lw=1
    plt.ylim([5e-23, 3e-19])
    plt.xlim([E_min, E_max])
    plt.xlabel('Energy [eV]', fontsize = fs)
    plt.ylabel(r'Cross section [m$^2$]', fontsize = fs)
    leg = plt.legend(fontsize = fs*0.9, ncol = 2, framealpha = 0.8)#, title = 'Interaction Type')

    plt.savefig(args.out_filepath + '/cross_sections_v6.pdf', format = 'pdf', bbox_inches="tight")