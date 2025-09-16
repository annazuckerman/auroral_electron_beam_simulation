import numpy as np
def patch_asscalar(a): # hack because of updated numpy deprecations
    return a.item()
setattr(np, "asscalar", patch_asscalar)
def patch_alen(a): # hack because of updated numpy deprecations
    return a.len()
setattr(np, "alen", patch_alen)
import pandas as pd
from astropy import units as u
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata 
from scipy.stats import binned_statistic
from scipy.special import expit
from scipy.integrate import quad 
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.integrate import trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy import stats
import seaborn as sns
import os
import scipy
import pickle


# define constants
c = 2.99e8 * u.m / u.s # speed of light [m/s]
c = c.to(u.m/u.s).value
m =  9.1093837e-31 * u.kg # e- mass [kg]
mH2 = 3.347649043E-27 # H2 mass [kg]
k = 1.380649e-23 # bolztman constant [J/k]
Rj = 71492e3 # Jupiter radius [m]
G = 6.6743e-11 # [Nm^2/kg^2]
e_ion = 15.43 # eV to ionize H2
stop_height_pcntle = 0.01 # % of ionizations occuring below stopping height by definition
Rj = 71492e3 # Jupiter radius [m]

def pick_theta_rutherford(E, m, cdf_val):
    '''
    Pick a theta value based on P(cos(scattering angle)) for a given energy, based on screened Rutherford
    formula from
    Eq. 4 in Lummerzheim 1989 
    Inputs:
        E (numpy array or float of length N): total energy(s) of incident electron(s)
    Returns:
        theta (numpy array of floats): scattering angles chosen
    '''
    epsilon = E/(m*c**2) # parentheses are unclear in the paper
    gamma_c = 0.6*E**(-0.09)
    gamma = gamma_c * 6.22e-5 / (epsilon*(epsilon + 2))
    N = 1 / (4*gamma*(1+gamma))
    cos_theta = - 1/(N*(2*cdf_val + (1/(2*N*(1+gamma))))) + 1 + 2*gamma
    return np.arccos(cos_theta)

def save_state_min_store(E_now, cos_theta, dt, ints, z_now, v_z_now, v_h_now, Ne, Ne_tot, i, N_oos_E, N_oos_z, path, filename):
    data = {'E_now':E_now, 'cos_theta':cos_theta, 'dt':dt, 'ints':ints, 'z_now':z_now, 'v_z_now':v_z_now, 'v_h_now':v_h_now,
               'i':i, 'Ne': Ne, 'Ne_tot': Ne_tot, 'N_oos_E': N_oos_E, 'N_oos_z':N_oos_z} 
    if not os.path.exists(path):
        os.mkdir(path)
    pickle.dump(data, open(path + '/' + filename, "wb" ))
    return

def save_results_min_store(z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, z_thermalization, E_thermalization, path, filename):
    arr = [z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, z_thermalization, E_thermalization]
    df = pd.DataFrame(arr)
    df.index = ['Ionization heights [m]', 'Rotational excitation heights [m]', 'Vibrational excitation heights [m]', 'B excitation heights [m]', 'C excitation heights [m]', 'a excitation heights [m]', 'b excitation heights [m]', 'c excitation heights [m]', 'e excitation heights [m]', 'Exit (altitude) heights [m]', 'Exit (energy) heights [m]', 'Exit (energy) energies [J]', 'Exit (altitude) energies [J]', 'Altitude thermalized electrons [m]', 'Energy thermalized electrons [J]']
    
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_hdf(path + '/' + filename, key='results', mode='w')
    return

def restore_state(filepath, run_ID):
    # read in the pickled state info
    state_dict = pickle.load(open(filepath + '/state_' + run_ID + '.pickle', "rb" ))
    E_now = state_dict['E_now']
    z_now = state_dict['z_now']
    v_z_now = state_dict['v_z_now']
    v_h_now = state_dict['v_h_now']
    dt = state_dict['dt']
    cos_theta = state_dict['cos_theta']
    ints = state_dict['ints']
    Ne = state_dict['Ne'] # current in sim
    Ne_tot = state_dict['Ne_tot'] # total in sim
    N_oos_z = state_dict['N_oos_z']
    N_oos_E = state_dict['N_oos_E']
    i = state_dict['i']
    # read the results hdf file with event altitudes
    results_df = pd.read_hdf(filepath + '/results_' + run_ID + '.h5', 'results')
    z_ion = list(results_df.loc['Ionization heights [m]'].dropna())
 #   z_el_scat = list(results_df.loc['Elastic scattering heights [m]'].dropna()) 
    z_rot_ex = list(results_df.loc['Vibrational excitation heights [m]'].dropna())
    z_vib_ex = list(results_df.loc['Rotational excitation heights [m]'].dropna()) 
    z_B_ex = list(results_df.loc['B excitation heights [m]'].dropna())
    z_C_ex = list(results_df.loc['C excitation heights [m]'].dropna())
    z_a_ex = list(results_df.loc['a excitation heights [m]'].dropna())
    z_b_ex = list(results_df.loc['b excitation heights [m]'].dropna())
    z_c_ex = list(results_df.loc['c excitation heights [m]'].dropna())
    z_e_ex = list(results_df.loc['e excitation heights [m]'].dropna())
    z_exit_z = list(results_df.loc['Exit (altitude) heights [m]'].dropna()) 
    z_exit_E = list(results_df.loc['Exit (energy) heights [m]'].dropna()) 
    E_exit_z = list(results_df.loc['Exit (energy) energies [J]'].dropna()) 
    E_exit_E = list(results_df.loc['Exit (altitude) energies [J]'].dropna())
    E_thermalization = list(results_df.loc['Energy thermalized electrons [J]'].dropna())
    z_thermalization = list(results_df.loc['Altitude thermalized electrons [m]'].dropna())
    return E_now, cos_theta, dt, ints, z_now, v_z_now, v_h_now, Ne, Ne_tot, z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_c_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, E_thermalization, z_thermalization, i, N_oos_E, N_oos_z

def construct_R_grid(z_grid, z_min, z_max, get_n_H2):
    '''
    Calculate the column mass density of H2 above the given heights by numerically integrating the density profile.
    For use as input to get_column_density() (see notes for that function). 
    Inputs:
        z_grid (numpy array): heights in meters to calculate column density for
    Returns:
        R (numpy array): array of numerically integrated column mass density values
    ''' 
    R_grid = np.zeros(len(z_grid))*np.nan
    for i in range(len(R_grid)):
        z_arr = np.linspace(z_grid[i], z_max, 1000)
        N = trapezoid(get_n_H2(z_arr), z_arr)   # quad(get_n_H2, z_grid[i], z_max)  
        R_grid[i] = N*mH2  
    return R_grid
    
def get_column_density(z, z_grid, R_grid):
    '''
    Calculate the column mass density of H2 above the given heights by interpolating a pre-computed array of
    column densities, computed by numerically integrating the density profile. This function should be used
    when the z array is very large or many arrays must be calculated in succession, so that one can call
    the function construct_R_grid() to construct the grid of R values for a grid of z values, and then use this 
    function to interpolate that grid for large numbers of heights.
    Inputs:
        z (numpy array): heights in meters to calculate column density for
        z_grid (numpy array): heights in meters
        R_grid (numpy array): column density values corresponding to z_grid
    Returns:
        R (numpy array): array of interpolated column mass density values
    '''  
    interpolate_R_of_z = spline(z_grid, R_grid)
    R = interpolate_R_of_z(z)
    return R

def get_z_from_column_density(R, z_grid, R_grid):
    '''
    Calculate the height corresponding to given column mass density of H2 by interpolating a pre-computed array of
    heights, computed by numerically integrating the density profile. This function should be used
    when the R array is very large or many arrays must be calculated in succession, so that one can call
    the function construct_R_grid() to construct the grid of R values for a grid of z values, and then use this 
    function to interpolate that grid for large numbers of R values.
    Inputs:
        R (numpy array): column density values to calculate height for
        z_grid (numpy array): heights [m]
        R_grid (numpy array): column densities corresponding to z_grid
    Returns:
        z (numpy array): array of interpolated heights [m]
    '''  
    interpolate_z_of_R = spline(R_grid[::-1], z_grid[::-1]) # make it so R is increasing
    z = interpolate_z_of_R(R)
    return z

def generate_rands(N):
    rand_nums = np.random.uniform(0,1,N) # will quickly run out with ionization e-
    N_rand = len(rand_nums)
    counter = 0
    return rand_nums, N_rand, counter

def ionization_xsec(E):
    '''
    Return ionization cross section for H2 with incident e- at given energies.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numy array): ionization cross section for H2, m^-2
    Note: according to NIST "the BEB model presented here is a nonrelativistic theory,
        and therefore should not be used for E > 10 keV."
    TO DO: Use given df/dw from the differntial xsec to calculate less approximate Q value
    '''
    
    R = 2.179873462921e-18  # [J]
    B = 2.472158546262e-18  # [J]
    U = 2.560278261132e-18  # [J]
    a0 = 5.29180000005e-11  # [m]

    N = 2 # (from NIST page)
    S = 4*np.pi*a0**2*N*(R/B)**2
    t = E/B 
    u = U/B
    # common assumptions for now:
    n = 1
    Q = 1
    
    T1 = Q*np.log(t)*(1-1/t**2)/2
    T2 = (2-Q)*(1 - 1/t - np.log(t)/(t+1))
    sigma = (S/(t + (u+1)/n)*(T1 + T2)) * np.heaviside(E-B, 0)
    
    sigma[t==0] = 0 # hacky way to get around nans from zero incident energy...
    
    return sigma    

def rot_excitation_xsec(E):
    '''
    NOTE USED
    Return rotational excitation cross section for H2 with incident e- at given energies.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    e0 = 4.38104E-02
    a0 = 1.00000E+00
    a1 = 2.10273E+00
    a2 = 1.21317E+04
    a3 = 5.07387E-04
    a4 = 1.10000E+00
    a5 = 1.10535E+03
    a6 = 0.00000E+00
    a7 = 1.50000E+00
   #         np.arctan((x/e0-a0)**a1 / a2) + a3*np.log(x/e0)) * (1.0/(x/e0)**a4) * (a5 + a6/(x/e0)**a7)
    sigma = (np.arctan((x/e0-a0)**a1 / a2) + a3*np.log(x/e0)) * (1.0/(x/e0)**a4) * (a5 + a6/(x/e0)**a7)
    sigma = sigma * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma * np.heaviside(x-e0, 0)

def rot_excitation_xsec_Jupiter(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=125K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    # weights from calculation of relative population of states of H2
    w = [1.0, 0.7419465413232091, 0.07563522168381312, 0.0016017922925646755] 
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def rot_excitation_xsec_482K(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=482K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    # weights from calculation of relative population of states of H2
    w = [0.47888426900533715, 1.0, 0.8075047731680346, 0.3812568510805055, 0.11506806845157908, 0.022979780181686484, 0.003088770086846174]
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def rot_excitation_xsec_500K(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=500K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    # weights from calculation of relative population of states of H2
    w = [0.4726785616352263, 1.0, 0.8288471139252584, 0.4069499366503302, 0.12940075628117637, 0.027583615489639283]
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def rot_excitation_xsec_900K(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=900K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    s5 = (np.arctan(((E_ev/0.154561)-1)**2.0183/637.099) + 0.00210037*np.log(E_ev/0.154561)) * (1/(E_ev/0.154561)**1.1) * (125.451) * np.heaviside(E_ev-0.154561, 0)
    s6 = (np.arctan(((E_ev/0.179595)-1)**2.00054/443.319) + 0.00262335*np.log(E_ev/0.179595)) * (1/(E_ev/0.179595)**1.1) * (103.024) * np.heaviside(E_ev-0.179595, 0)
    s7 = (np.arctan(((E_ev/0.203269)-1)**1.98407/328.089) + 0.00318164*np.log(E_ev/0.203269)) * (1/(E_ev/0.203269)**1.1) * (87.8482) * np.heaviside(E_ev-0.203269, 0)
    s8 = (np.arctan(((E_ev/0.225038)-1)**1.9727/258.185) + 0.00392664*np.log(E_ev/0.225038)) * (1/(E_ev/0.225038)**1.1) * (77.0872) * np.heaviside(E_ev-0.225038, 0)
    s9 = (np.arctan(((E_ev/0.245447)-1)**1.96188/210.117) + 0.00468342*np.log(E_ev/0.245447)) * (1/(E_ev/0.245447)**1.1) * (69.0257) * np.heaviside(E_ev-0.245447, 0)
    s10 = (np.arctan(((E_ev/0.264223)-1)**1.95143/176.049) + 0.00540323*np.log(E_ev/0.264223)) * (1/(E_ev/0.264223)**1.1) * (62.8829) * np.heaviside(E_ev-0.264223, 0)
    # weights from calculation of relative population of states of H2
    w = [0.3579661237893469, 0.8844892100579985, 1.0, 0.7821969214181066, 0.46278363600128974, 0.21437539982516335, 0.0790863892242686, 0.023461465873987352, 0.005630508910132541]
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5 + w[5]*s6 + w[6]*s7 + w[7]*s8 + w[8]*s9) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def rot_excitation_xsec_1400K(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=1400K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    s5 = (np.arctan(((E_ev/0.154561)-1)**2.0183/637.099) + 0.00210037*np.log(E_ev/0.154561)) * (1/(E_ev/0.154561)**1.1) * (125.451) * np.heaviside(E_ev-0.154561, 0)
    s6 = (np.arctan(((E_ev/0.179595)-1)**2.00054/443.319) + 0.00262335*np.log(E_ev/0.179595)) * (1/(E_ev/0.179595)**1.1) * (103.024) * np.heaviside(E_ev-0.179595, 0)
    s7 = (np.arctan(((E_ev/0.203269)-1)**1.98407/328.089) + 0.00318164*np.log(E_ev/0.203269)) * (1/(E_ev/0.203269)**1.1) * (87.8482) * np.heaviside(E_ev-0.203269, 0)
    s8 = (np.arctan(((E_ev/0.225038)-1)**1.9727/258.185) + 0.00392664*np.log(E_ev/0.225038)) * (1/(E_ev/0.225038)**1.1) * (77.0872) * np.heaviside(E_ev-0.225038, 0)
    s9 = (np.arctan(((E_ev/0.245447)-1)**1.96188/210.117) + 0.00468342*np.log(E_ev/0.245447)) * (1/(E_ev/0.245447)**1.1) * (69.0257) * np.heaviside(E_ev-0.245447, 0)
    s10 = (np.arctan(((E_ev/0.264223)-1)**1.95143/176.049) + 0.00540323*np.log(E_ev/0.264223)) * (1/(E_ev/0.264223)**1.1) * (62.8829) * np.heaviside(E_ev-0.264223, 0)
    # weights from calculation of relative population of states of H2
    w = [0.2907715776465393, 0.7700150646302739, 1.0, 0.9629551906904974, 0.751717403610959, 0.49241905706885575, 0.27532243698469894, 0.13266965864485883, 0.05542904400167633, 0.02015933474537375, 0.006400325081107996]
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5 + w[5]*s6 + w[6]*s7 + w[7]*s8 + w[8]*s9 + w[9]*s10 ) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def rot_excitation_xsec_2000K(E):
    '''
    Return rotational excitation cross section for H2 with incident e- at given energies,
    for body with T=2000K.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E_ev = E * 6.242e+18 # convert to eV
    # Rotational excitation cross sections from MCCC database fit and fit params, for transitions J=0--> through J=3-->5
    # Using all transitions starting from a state with >1% of H2 population
    s1 = (np.arctan(((E_ev/0.0438104)-1)**2.10273/12131.7) + 0.000507387*np.log(E_ev/0.0438104)) * (1/(E_ev/0.0438104)**1.1) * (1105.35) * np.heaviside(E_ev-0.0438104, 0)
    s2 = (np.arctan(((E_ev/0.0726544)-1)**2.08001/3802.07) + 0.000821337*np.log(E_ev/0.0726544)) * (1/(E_ev/0.0726544)**1.1) * (380.284) * np.heaviside(E_ev-0.0726544, 0)
    s3 = (np.arctan(((E_ev/0.100954)-1)**2.05379/1734.98) + 0.0010799*np.log(E_ev/0.100954)) * (1/(E_ev/0.100954)**1.1) * (227.183) * np.heaviside(E_ev-0.100954, 0)
    s4 = (np.arctan(((E_ev/0.128438)-1)**2.03567/988.085) + 0.00157183*np.log(E_ev/0.128438)) * (1/(E_ev/0.128438)**1.1) * (161.256) * np.heaviside(E_ev-0.128438, 0)
    s5 = (np.arctan(((E_ev/0.154561)-1)**2.0183/637.099) + 0.00210037*np.log(E_ev/0.154561)) * (1/(E_ev/0.154561)**1.1) * (125.451) * np.heaviside(E_ev-0.154561, 0)
    s6 = (np.arctan(((E_ev/0.179595)-1)**2.00054/443.319) + 0.00262335*np.log(E_ev/0.179595)) * (1/(E_ev/0.179595)**1.1) * (103.024) * np.heaviside(E_ev-0.179595, 0)
    s7 = (np.arctan(((E_ev/0.203269)-1)**1.98407/328.089) + 0.00318164*np.log(E_ev/0.203269)) * (1/(E_ev/0.203269)**1.1) * (87.8482) * np.heaviside(E_ev-0.203269, 0)
    s8 = (np.arctan(((E_ev/0.225038)-1)**1.9727/258.185) + 0.00392664*np.log(E_ev/0.225038)) * (1/(E_ev/0.225038)**1.1) * (77.0872) * np.heaviside(E_ev-0.225038, 0)
    s9 = (np.arctan(((E_ev/0.245447)-1)**1.96188/210.117) + 0.00468342*np.log(E_ev/0.245447)) * (1/(E_ev/0.245447)**1.1) * (69.0257) * np.heaviside(E_ev-0.245447, 0)
    s10 = (np.arctan(((E_ev/0.264223)-1)**1.95143/176.049) + 0.00540323*np.log(E_ev/0.264223)) * (1/(E_ev/0.264223)**1.1) * (62.8829) * np.heaviside(E_ev-0.264223, 0)
    s11 = (np.arctan(((E_ev/0.296604)-1)**1.93208/132.803) + 0.00666411*np.log(E_ev/0.296604)) * (1/(E_ev/0.296604)**1.1) * (54.3727)
    s12 = (np.arctan(((E_ev/0.31048200000000004)-1)**1.92635/119.406) + 0.00744393*np.log(E_ev/0.31048200000000004)) * (1/(E_ev/0.31048200000000004)**1.1) * (51.2891)   
    # weights from calculation of relative population of states of H2
    w = [0.2412307682894333, 0.6631811365989436, 0.9281909599735108, 1.0, 0.9066868340480482, 0.7161409946349029, 0.5012085399780347, 0.31384400623567776, 0.176887515178004, 0.09009604995039368, 0.04158661238685897, 0.01743100372720285, 0.006644664970690637]
    w = w / np.sum(w)
    sigma = (w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5 + w[5]*s6 + w[6]*s7 + w[7]*s8 + w[8]*s9 + w[9]*s10 ) * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def elastic_scat_xsec(E):
    '''
    Return elastic scattering cross section (rotationally elastic, H2 (X 1Σg+, vi = 0, Ni = 0)  →  H2 (X 1Σg+, vf = 0, Nf = 0))
    for H2 with incident e- at given energies.
    From fitting data from Scarlett mccc-db.org database using functional form provided.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    a0 = 0.00000E+00
    a1 = 1.37405E+00
    a2 = 9.50341E+00
    a3 = 0.00000E+00
    a4 = 1.20000E+00
    a5 = 4.28473E+02
    a6 = 0.00000E+00
    a7 = 1.50000E+00
    sigma = (np.arctan((x-a0)**a1 / a2 ) + a3*np.log(x)) * (1.0/x**a4) * (a5 + a6/x**a7)
    sigma = sigma * (5.2918e-11)**2 # convert a0^2 to m^2
    return sigma 

def B_excitation_xsec(E):
    '''
    Return cross section for the H2(X1Sg,vi=0) -> H2(B1Su) transition at given energies.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    e0 = 1.11829E+01 #eV
    a0 = 1.20670709e-10
    a1 = 4.86253324e-21
    a2 = -5.55016345e-20
    a3 = 1.39389371e-19
    a4 = -1.24028577e-19
    a5 = 3.46525815e-20
    x = x/e0
    sigma = np.abs(((x-1)/x) * (a0**2/x * np.log(x) + a1/x + a2/x**2 + a3/x**3 + a4/x**4 + a5/x**5))
    #sigma[E==0] = 0 # hacky way to get around nans from zero incident energy...
    return sigma * np.heaviside(x-1, 0)

def C_excitation_xsec(E):
    '''
    Return cross section for the H2(X1Sg,vi=0) -> H2(C1Pu) transition at given energies.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    e0 = 1.22910E+01 #eV
    a0 = 1.10548818e-10
    a1 = 1.27577809e-20
    a2 = -9.29965508e-20
    a3 = 2.16793338e-19
    a4 = -2.05316538e-19
    a5 = 6.94526275e-20
    x = x/e0
    
    sigma = np.abs(((x-1)/x) * (a0**2/x * np.log(x) + a1/x + a2/x**2 + a3/x**3 + a4/x**4 + a5/x**5))  
    #sigma[E==0] = 0 # hacky way to get around nans from zero incident energy...
    return sigma * np.heaviside(x-1, 0)

def a_excitation_xsec(E):
    '''
    Return cross section for the e + H2(X1Sg,vi=0) -> e + H2(a3Sg) transition.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E = E * 6.242e+18 # convert to eV
    e0 = 1.17934E+01 # given on datafile page # 12.156935469153458 
    a0 = 0.004654237310341429
    a1 = 39.19641746862529
    a2 = -156.255997891616
    a3 = 204.52585973437647
    a4 = -81.09095833035742
    a5 = 2.973371006682526
    a6 = 0.011605495479858084
    x = E/e0
    X = 20 # peicewise junction point in eV
    sigma =  np.abs((x-1)/x * (a0**2/x + a1/x**2 + a2/x**3 + a3/x**4 + a4/x**5)) * np.heaviside(x-1,0) * np.heaviside(X-E,0) + (1/(x+a6)**a5) * np.heaviside(E-X,0) 
    return sigma * (5.2918e-11)**2 # convert to m^2

def b_excitation_xsec(E):
    '''
    Return cross section for the e + H2(X1Sg,vi=0) -> e + H2(b3Su) (dissociative excitation) transition.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    e0 = 4.47713E+00
    a0 = 9.83930E+00
    a1 = 1.39710E+00
    a2 = 4.07460E+00
    a3 = 2.99510E+00
    x = x/e0
    sigma = a0 * (x-1)**(-a1**2) * np.exp(-a2/(x-1)**a3) * np.heaviside(x-1, 0)
    sigma[np.isnan(sigma)] = 0.0 # we want sigma(E<e0) to be zero, not Nan
    return sigma * (5.2918e-11)**2 # convert a0^2 to m^2

def c_excitation_xsec(E):
    '''
    Return cross section for the e + H2(X1Sg,vi=0) -> e + H2(c3Pu) transition.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    e0 = 12.156935469153458
    a0 = 0.0001057547205682496
    a1 = 0.13244429434376978
    a2 = 3.654004835819313
    a3 = -1.78274448372897
    a4 = 2.979627095789858
    x = x/e0
    sigma = np.abs((x-1)/x * (a0**2/x + a1/x**2 + a2/x**3 + a3/x**4 + a4/x**5)) * np.heaviside(x-1,0)
    return sigma * (5.2918e-11)**2 # convert to m^2

def d_excitation_xsec(E):
    '''
    Return cross section for the e + H2(X1Sg,vi=0) -> e + H2(d3Pu) transition.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    E = E * 6.242e+18 # convert to eV
    e0 = 1.38553E+01 # given on datafile page # 12.156935469153458 
    a0 = 0.20448566328198925
    a1 = 37.19730746608392
    a2 = -144.8282323180027
    a3 = 189.04235352935785
    a4 = -81.27277434080747
    a5 = 2.936740712680107
    a6 = 0.7933973571033605
    x = E/e0
    X = 20 # peicewise junction point in eV
    sigma =  np.abs((x-1)/x * (a0**2/x + a1/x**2 + a2/x**3 + a3/x**4 + a4/x**5)) * np.heaviside(x-1,0) * np.heaviside(X-E,0) + (1/(x+a6)**a5) * np.heaviside(E-X,0) 
    return sigma * (5.2918e-11)**2 # convert to m^2

def e_excitation_xsec(E):
    '''
    Return cross section for the e + H2(X1Sg,vi=0) -> e + H2(e3Su) transition.
    From fitting data from MCCC database (Scarlett et. al.) using provided functional form.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    # sigma in m^2
    E = E * 6.242e+18 # convert to eV
    e0 = 1.38553E+01 # given on datafile page # 12.156935469153458 
    a0 = 9.248766122881518
    a1 = -412.7369902768685
    a2 = 742.9821741190286
    a3 = -590.2249877716694
    a4 = 175.63369572811268
    a5 = 3.100934937938628
    a6 = 0.7718889136087155
    x = E/e0
    X = 20 # peicewise junction point in eV
    sigma =  np.abs((x-1)/x * (a0**2/x + a1/x**2 + a2/x**3 + a3/x**4 + a4/x**5)) * np.heaviside(x-1,0) * np.heaviside(X-E,0) + (1/(x+a6)**a5) * np.heaviside(E-X,0) 
    return sigma * (5.2918e-11)**2 # convert to m^2
    
def vib_excitation_xsec(E):
    '''
    Return cross section for vibrational excitaion (v= 0->1) transition at given energies.
    From fitting data from Table 4 of Yoon et al. 2008 using a functional form from Scarlett et al.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    WARNING: Only valid for E >= 1ev (these electrons should have exited the simulation)
    '''
    x = E * 6.242e+18 # convert to eV
    a0 = -1.00041903e-06
    a1 = 4.38456614e-02
    a2 = 2.57283470e+01
    a3 = -8.88972971e+01
    a4 = 1.05467382e+02
    a5= -4.12781843e+01
    e0 = 8.77596948e-01
    x = x/e0
    conversion = 10**(-16) / (1e4) # convert to m^2
    return (np.abs(((x-1)/x) * (a0**2/x * np.log(x) + a1/x + a2/x**2 + a3/x**3 + a4/x**4 + a5/x**5))) * conversion * np.heaviside(x-1, 0)
    
def vahedi_ejected_energy(E_inc, E_ion):
    '''
    Pick the energy of the ejected (created) electron during ionization, using the method given in Vahedi and Surendra 
    with B(E_inc) value from Opal 1971.
    Inputs:
        E_inc (numpy array): energies of incident electrons (J)
        E_ion (float):       ionization energy of the molecule (J)
    Returns:
        E_ej (numpy array): energies of ejected electrons (J)
    TO DO: streamline unit conversions, and make a new function using the NIST diff xsec if possible
    '''
    def B(E_inc):
        return 8.3 * np.ones(len(E_inc)) # eV
    E_inc = E_inc * 6.242e+18 # convert to eV, since Vahedi formula constants us eV
    E_ion = E_ion * 6.242e+18 # convert to eV, since Vahedi formula constants us eV
    R = np.random.uniform(0,1,len(E_inc))
    E_ej = B(E_inc) * np.tan(R * np.arctan((E_inc - E_ion)/(2*B(E_inc))))
    return E_ej / 6.242e+18 # convert to J

def get_uncertainties(z_arr, nfolds, bins, Ntot, stop_height_pcntle, get_n_H2, get_pressure, z_max, atm_type, min_val, max_val, make_plot = False, make_subset_plot = False, npoints = 100):
    '''
    Calculate uncertainty on the value in each histogram bin by n-fold boostrapping. Calculate the histogram nfolds times,
    and take the standard deviation in bin heights over all folds as the uncertianty in that bin. 
    Also calculate uncertainty on R0 and peak location. 
    Inputs:
     z_arr (numpy array of floats): array of heights where the interaction occured
     nfolds (int): how many folds to calculate the histogram over
     bins (numpy array of floats): bins to calculate histogram over
     Ntot (int): number of incident electrons used in the simulation
     stop_height_pcntle (float): what percentile to use when defining stopping height
     get_n_H2 (function): the function to use for column density (depends on chosen atmospheric density profile)
     get_pressure (function) :the function to use for pressure (depends on chosen atmospheric density profile)
     z_max (float): maximum height used in simulation to integrated column density to (m)
     atm_type (str): which atmosphere to use
     make_subset_plot (bool): whether to make and save another plot including the subsampled histograms
     npoints (int): how many points to calculate the kde curve over
    Returns:
     bins_err (numpy array of floats): array of uncertainty values on histogram bins
     z_peak_mean (float): peak locatin (m)
     z_peak_err (float): uncertainty on peak location (m)
     P_peak_mean (float): uncertainty on peak location (Pa)
     P_peak_err (float): uncertainty on peak location (Pa)
     R0_sd (float): uncertainty on R0
     Ro_mean (float):
     stopping_P_mean (float):
     stopping_P_sd (float):
     RoverR0_peak_sd (float):
     RoverR0_peak_mean (float):
     R_peak_sd, R_peak_mean (float):
     R_peak_sd, R_peak_mean (float): 
    '''
    if make_plot:    
        print('making full histogram plot')
        fig1 = plt.figure(figsize = [8,5]) 
        full_hist, bins = np.histogram(z_arr, bins = bins)
        plt.step(bins[:-1]/1000, full_hist, alpha = 1, color = 'C1', linewidth = 0.5)
    
    z_arr = z_arr[np.invert(np.isnan(z_arr))]
    N = len(z_arr)
    N_per_fold = int(N/nfolds)
    print('calculating uncertainties in histogram bins with:')
    print('    ', nfolds, ' folds, ', N_per_fold, ' per fold, of', N, ' total interactions.')
    
    all_folds_binned = np.zeros([nfolds, len(bins)-1])*np.nan
    z_remaining = np.copy(z_arr)
    all_z_peaks = np.zeros([nfolds])*np.nan
    all_P_peaks = np.zeros([nfolds])*np.nan
    all_RoverR0_peaks = np.zeros([nfolds])*np.nan
    all_R_peaks = np.zeros([nfolds])*np.nan
    all_R0 = np.zeros([nfolds])*np.nan
    all_stopping_P = np.zeros([nfolds])*np.nan
    for i in range(nfolds):
        #np.random.seed(42)
        fold = np.random.choice(z_remaining, N_per_fold, replace = True) # choose with replacement        
        # collect bin values
        counts, bins = np.histogram(fold, bins = bins)
        all_folds_binned[i,:] = counts
        
        # collect R0 values
        stopping_height = np.percentile(fold, stop_height_pcntle)
        stopping_P = get_pressure(stopping_height)
        all_stopping_P[i] = stopping_P
        N0, err = quad(get_n_H2, stopping_height, z_max)
        R0 = N0*mH2 # column density at stopping height
        all_R0[i] = R0
        
        # collect z-peak values
        z_peak = get_kde_values(fold, bins, Ntot, atm_type, get_n_H2, min_val, max_val, npoints, make_plot = False)[1]
        P_peak = get_pressure(z_peak)
        z_grid = np.linspace(0,max_val,10000)
        R_grid = construct_R_grid(z_grid, min_val, max_val, get_n_H2)
        R_peak = get_column_density(z_peak, z_grid, R_grid) 
        all_R_peaks[i] = R_peak/R0
        all_RoverR0_peaks[i] = R_peak/R0
        all_z_peaks[i] = z_peak
        all_P_peaks[i] = P_peak
        
        if make_subset_plot:
            print('making subset histogram plot')
            plt.step(bins[:-1]/1000, counts, alpha = 0.1, color = 'C0', linewidth = 2)
            
    sd = np.nanstd(all_folds_binned, axis = 0)
    sd_tot = np.sqrt(nfolds) * sd   # err on each bin is sqrt(sum from 0 to nfolds of 
    R0_sd = np.nanstd(all_R0)
    R0_mean = np.nanmean(all_R0)
    z_peak_sd = np.nanstd(all_z_peaks)
    z_peak_mean = np.nanmean(all_z_peaks)
    P_peak_sd = np.nanstd(all_P_peaks)
    P_peak_mean = np.nanmean(all_P_peaks)
    RoverR0_peak_sd = np.nanstd(all_RoverR0_peaks)
    RoverR0_peak_mean = np.nanmean(all_RoverR0_peaks)
    stopping_P_sd = np.nanstd(all_stopping_P)
    stopping_P_mean = np.nanmean(all_stopping_P)
    R_peak_sd = np.nanstd(all_R_peaks)
    R_peak_mean = np.nanmean(all_R_peaks)

    if make_subset_plot:
        print('making subset histogram plot')
        plt.plot([],[], alpha = 0.1, color = 'C0', linewidth = 2, label = 'subsampled histograms')
        plt.plot([],[], alpha = 1, color = 'C1', linewidth = 0.5, label = 'full histogram')
        plt.plot([],[], alpha = 0.3, color = 'C3', linewidth = 4, label = '3*SD uncertainty')
        plt.legend()
        plt.xlabel('Altitude [km]')
        plt.ylabel('Counts (no normalization)')
        plt.yscale('log')
        binwidth = (bins[1]-bins[0])
        bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth
        mean = np.nanmean(all_folds_binned, axis = 0)
        plt.fill_between(bincenters/1000, mean-3*sd, mean+3*sd, alpha = 0.3, color = 'C3')
        plt.fill_between(bincenters/1000, full_hist-3*sd_tot, full_hist+3*sd_tot, alpha = 0.3, color = 'C3')
        plt.savefig('./test_plots/bootstrap_histograms2.png', dpi = 300)
        plt.close(fig1)
    
    bins_sd = sd_tot / Ntot # normalizing just divides the resulting SD 
    return bins_sd, z_peak_sd, z_peak_mean, P_peak_sd, P_peak_mean, RoverR0_peak_sd, RoverR0_peak_mean, R_peak_sd, R_peak_mean, R0_sd, R0_mean, stopping_P_sd, stopping_P_mean

def get_uncertainties_general(events_list, nfolds, bins, Ntot, atm_type, min_val, max_val, get_nH2, plot_name = None, make_plot = False, make_subset_plot = False, return_kde_curve_median_and_sd = False, plot_each_kde = False, npoints = 100):
    '''
    Calculate uncertainty on the value in each histogram bin by n-fold boostrapping. Calculate the histogram nfolds times,
    and take the standard deviation in bin heights over all folds as the uncertianty in that bin. 
    Also calculate uncertainty on peak location and value. 
    Inputs:
     events_list (numpy array of floats): array of events to histogram
     nfolds (int): how many folds to calculate the histogram over
     bins (numpy array of floats): bins to calculate histogram over
     Ntot (int): number of incident electrons used in the simulation
     atm_type (str): which atmosphere to use
     make_subset_plot (bool): whether to make and save another plot including the subsampled histograms
     min_val (float): minimum value to calculate kde curve over
     max_val (float):maximum value to calculate kde curve over
     get_nH2 (function): number density profile [#/m^3] function of z[m]
     return_kde_curve_median_and_sd (bool): whether or not to return the median and sd of kde curve values
     npoints (int): how many points to calculate the kde curve over
    Returns:
     bins_err (numpy array of floats): array of uncertainty values on histogram bins
     peak_loc_mean (float): peak location (same units as events_list)
     peak_loc_err (float): uncertainty on peak location (same units as events_list)
     peak_mean (float): uncertainty on peak value
     peak_err (float): uncertainty on peak value
     return_kde_curve_median (numpy array):
     return_kde_curve_sd (numpy array):
    '''
    # make full histogram
    if make_plot:    
        print('making full histogram plot')
        fig1 = plt.figure(figsize = [8,5]) 
        full_hist, bins = np.histogram(events_list, bins = bins)
        plt.step(bins[:-1]/1000, full_hist, alpha = 1, color = 'C1', linewidth = 0.5)
    if plot_each_kde:
        fig1 = plt.figure(figsize = [8,5])
        plt.xlabel('KDE curve value')
        plt.ylabel('Altitude [km]')
    
    # remove nans
    events_list = events_list[np.invert(np.isnan(events_list))]
    N = len(events_list)
    N_per_fold = int(N/nfolds)
    print('calculating uncertainties in histogram bins with:')
    print('    ', nfolds, ' folds, ', N_per_fold, ' per fold, of', N, ' total interactions.')
    
    # do the bootstrapping
    events_remaining = np.copy(events_list)
    all_folds_binned = np.zeros([nfolds, len(bins)-1])*np.nan
    all_peak_vals = np.zeros([nfolds])*np.nan
    all_peak_locs = np.zeros([nfolds])*np.nan
    if return_kde_curve_median_and_sd:
        all_kde_curve_values = np.zeros([nfolds, npoints])*np.nan
    for i in range(nfolds):
        fold = np.random.choice(events_remaining, N_per_fold, replace = True) # choose with replacement
        
        # collect bin values
        counts, bins = np.histogram(fold, bins = bins)
        all_folds_binned[i,:] = counts
        
        # collect z-peak values
        if return_kde_curve_median_and_sd:
            peak_val, peak_loc, kde_curve = get_kde_values(fold, bins, Ntot, atm_type, get_nH2, min_val, max_val, npoints, make_plot = False, return_kde_curve = True)
        else:
            peak_val, peak_loc = get_kde_values(fold, bins, Ntot, atm_type, get_nH2, min_val, max_val, npoints, make_plot = False, return_kde_curve = False)
        all_peak_locs[i] = peak_loc
        all_peak_vals[i] = peak_val
        if return_kde_curve_median_and_sd:
            all_kde_curve_values[i,:] = kde_curve
        
        if make_subset_plot:
            plt.step(bins[:-1]/1000, counts, alpha = 0.1, color = 'C0', linewidth = 2)
        if plot_each_kde:
            plt.plot(kde_curve, np.linspace(min_val,max_val,npoints)/1e3, color = 'C0', alpha = 0.25)
            
    sd = np.nanstd(all_folds_binned, axis = 0) # standard deviation on bin counts
    sd_tot = np.sqrt(nfolds) * sd   # err on each bin is sqrt(sum from 0 to nfolds of 
    peak_loc_sd = np.nanstd(all_peak_locs)
    peak_loc_mean = np.nanmean(all_peak_locs)
    peak_val_sd = np.nanstd(all_peak_vals)
    peak_val_mean = np.nanmean(all_peak_vals)
    if return_kde_curve_median_and_sd:
        kde_curve_median = np.nanmedian(all_kde_curve_values, axis = 0)
        kde_curve_sd = np.nanstd(all_kde_curve_values, axis = 0)
  
    if make_subset_plot:
        print('making subset histogram plot')
        plt.plot([],[], alpha = 0.1, color = 'C0', linewidth = 2, label = 'subsampled histograms')
        plt.plot([],[], alpha = 1, color = 'C1', linewidth = 0.5, label = 'full histogram')
        plt.plot([],[], alpha = 0.3, color = 'C3', linewidth = 4, label = '3*SD uncertainty')
        plt.legend()
        plt.xlabel('Altitude [km]')
        plt.ylabel('Counts (no normalization)')
        plt.yscale('log')
        binwidth = (bins[1]-bins[0])
        bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth
        mean = np.nanmean(all_folds_binned, axis = 0)
        plt.fill_between(bincenters/1000, mean-3*sd, mean+3*sd, alpha = 0.3, color = 'C3')
        plt.fill_between(bincenters/1000, full_hist-3*sd_tot, full_hist+3*sd_tot, alpha = 0.3, color = 'C3')
        plt.savefig('./Analysis_Plots/bootstrap_histograms'+plot_name+'.png', dpi = 300)
        plt.close(fig1)
        
    if plot_each_kde:
        plt.plot(kde_curve_median, np.linspace(min_val,max_val,npoints)/1e3, color = 'k', alpha = 1.0, label = 'Median')
        plt.plot(kde_curve_median - kde_curve_sd, np.linspace(min_val,max_val,npoints)/1e3, ls = 'dashed', color = 'k', alpha = 1.0, label = '+/- SD')
        plt.plot(kde_curve_median + kde_curve_sd, np.linspace(min_val,max_val,npoints)/1e3, ls = 'dashed', color = 'k', alpha = 1.0)
        plt.xlim([1e-7,9e-4])
        plt.semilogx()
        plt.legend()
        plt.savefig('./Analysis_Plots/bootstrap_kde_curves'+plot_name+'.png', dpi = 300)
        plt.close(fig1)
    
    bins_sd = sd_tot / Ntot # normalizing just divides the resulting SD 
    if return_kde_curve_median_and_sd:
        return bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean, kde_curve_median, kde_curve_sd
    else:
        return bins_sd, peak_loc_sd, peak_loc_mean, peak_val_sd, peak_val_mean


def get_kde_values(arr, bins, Ntot, atm_type, get_nH2, min_val, max_val, n_points, make_plot = False, return_kde_curve = False):
    '''
    Calculate uncertainty on peak location and stopping column density by smoothing histograms with Kernel Density Estimation
    and taking the standard deviation in parameters with n-fold boostrapping.
    and takin
    Inputs:
     z_arr (numpy array of floats): array of heights where the interaction occured
     bins (numpy array of floats): bins to calculate histogram over (only for plotting)
     Ntot (int): number of incident electrons used in the simulation
     atm_type (str): which profile to use
     make_plot (bool): whether to plot the kde results
     min_val (float): minimum value to calculate kde curve over
     max_val (float):maximum value to calculate kde curve over
     get_nH2 (function): number density profile [#/m^3] function of z[m]
     n_points (int): number of points to evaluate kde curve at
     return_kde_curve (bool): whether to return the values of the kde curve
    Returns:
     peak_val (float): peak value [units of z_arr]    
     peak_loc (float): peak location [m]
    '''           
    kde = stats.gaussian_kde(arr)
    stats_kde = np.linspace(min_val,max_val,n_points)
    stats_kde_curve = kde(stats_kde)
    stats_kde_max = np.max(stats_kde_curve)
    stats_kde_max_loc = stats_kde[stats_kde_curve == stats_kde_max]  
    
    peak_loc = stats_kde_max_loc
    peak_val = stats_kde_max
   
    if make_plot:
        print('making kde plot')
        binwidth = (bins[1]-bins[0])
        bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth

        fig1 = plt.figure(figsize = [8,5], dpi = 200) 
        ax1 = plt.gca()
        full_hist, bins = np.histogram(z_arr, bins = bins)
        hist_max = np.max(full_hist)
        ax1.step(bins[:-1]/1000, full_hist/hist_max , alpha = 1, linewidth = 0.5, label = 'histogram at bins[:-1]')
        ax1.step(bins[1:]/1000, full_hist/hist_max , alpha = 1, linewidth = 0.5, label = 'histogram at bins[1:]')
        ax1.step(bincenters/1000, full_hist/hist_max , alpha = 1, linewidth = 0.5, label = 'histogram at bincenters')
        ax1.step(bincenters/1000, full_hist/hist_max , alpha = 1, linewidth = 0.5, label = 'histogram at bincenters, step = mid', where = 'mid')

        ax1.plot(stats_kde_z/1000, stats_kde_curve/stats_kde_max, label = 'scipy kde smoothing') 
        ylims = ax1.get_ylim()
        ax1.vlines([stats_kde_max_loc/1000], ymin = ylims[0], ymax = ylims[1], ls = 'dashed', color = 'k', alpha = 0.6, label = 'scipy kde peak')
    
        ax1.legend()
        ax1.set_ylabel('Normalized frequency')
        ax1.set_xlabel('Height [km]')
        fig1.savefig('./test_plots/kde_histograms_test6.png')
        plt.close(fig1)

    if return_kde_curve:
         return peak_val, peak_loc, stats_kde_curve
    else:
        return peak_val, peak_loc
        
def get_Hiraki_parameterization_curve(z, z_min, z_max, e0, get_nH2):
    '''
    Return Hiraki paramerterization curve.
    Inputs:
     z (numpy array of floats): array of heights where the interaction occured [m]
     z_min (float): minimum z for calculating curve [m]
     z_max (float): maximum z for calculating curve [m]
     e0 (float): incident beam energy [ev]
     stopping_height (m): stopping height
    Returns:
     q (numpy array of floats): Hiraki parameterization of q_ion
    '''
    R = np.zeros(len(z))
    for i in range(len(R)):
        zi = z[i]
        N, err = quad(get_nH2, zi, z_max) # #/m^2
        Ri = N*mH2 # column density kg/m^2
        R[i] = Ri

    e_ion = 30 #15.43 # ev from Hiraki
    rho = get_nH2(z)*mH2 
    R0 = 3.39e-5 * (e0/1000)**1.39 #Hiraki
    #R0 = quad(n_H2_Jupiter_Hiraki, stopping_height, z_max)[0]*mH2 # by eye (same order of magnitude as Hiraki)
    k = 0.13 + 0.89*(1-1.1*np.tanh(np.log10(e0/1000) -1))
    def get_lam0(x):
        lam0 = np.zeros(len(x))
        r1 = (x>=0) * (x<=0.3)
        r2 = (x>0.3) * (x<=0.825)
        r3 = (x>0.825) * (x<=1)
        lam0[r1] = -669.53*x[r1]**4 + 536.18*x[r1]**3 - 159.86*x[r1]**2 + 18.586*x[r1] + 0.506
        lam0[r2] = 0.767*x[r2]**4 - 5.9034*x[r2]**3 + 12.119*x[r2]**2 - 9.734*x[r2] + 2.7470
        lam0[r3] = -0.8091*x[r3]**3 + 2.4516*x[r3]**2 - 2.4777*x[r3] + 0.8353     
        return lam0

    lam = get_lam0(R/R0) * k
    q = (e0/e_ion)*(rho/R0)*lam # #/m
    return q  

def construct_profiles(atm_type, z_max, sonora_filepath):
    '''
    Read in Sonora model ouputs and construct spline interpolation functions for nH2(z),
    P(z), and z(P). Spline interpolation has been shown to be almost as fast, and
    more accurate, compared with fitting functional forms for these quantities.
    a polynomial fit function for these 
    Inputs:
        z_max (float)): height at which electrons are removed from simulation [m]
        atm_type (str): which object to use
    Returns:
        get_nH2 (function): spline interpolation of ln(nH2(z)) [m^-3]
        get_P (function): spline interpolation of ln(P(z)) [Pa]
    ''' 
 
    if atm_type == 'Jupiter':
        T_TOA_sonora = 150.78 # K
        g = 24.79 # m/s^2
        R = Rj
        M = g*R**2/G 
        filename = 'jupiter_1e-8_final.pkl'
        df = pd.read_pickle(sonora_filepath + '/' + filename)
    elif atm_type == 'T900_g5.0':
        T_TOA_sonora = 283.47 # K
        g = 1000 # m/s^2
        R = Rj
        M = g*R**2/G
        filename = atm_type+'_nc_moist_0.0metal_NR_smart.atm'
        df = pd.read_csv(sonora_filepath + '/' + atm_type + '_nc_moist_0.0metal_NR_smart.atm', sep='\s+', skiprows = 0)
    elif atm_type == 'T1400_g4.0':
        T_TOA_sonora = 550.72 # K
        g = 100 # m/s^2# K
        R = Rj
        M = g*R**2/G
        filename = atm_type+'_nc_moist_0.0metal_NR_smart.atm'
        df = pd.read_csv(sonora_filepath + '/' + atm_type + '_nc_moist_0.0metal_NR_smart.atm', sep='\s+', skiprows = 0)
    elif atm_type == 'T900_g4.0':
        T_TOA_sonora =  263.422216 # K
        g = 100 # m/s^2# K
        R = Rj
        M = g*R**2/G
        filename = atm_type+'_nc_moist_0.0metal_NR_smart.atm'
        df = pd.read_csv(sonora_filepath + '/' + atm_type + '_nc_moist_0.0metal_NR_smart.atm', sep='\s+', skiprows = 0)
    elif atm_type == 'T1400_g5.0':
        T_TOA_sonora = 575.509596 # K
        g = 1000 # m/s^2
        R = Rj
        M = g*R**2/G
        filename = atm_type+'_nc_moist_0.0metal_NR_smart.atm'
        df = pd.read_csv(sonora_filepath + '/' + atm_type + '_nc_moist_0.0metal_NR_smart.atm', sep='\s+', skiprows = 0)
    elif atm_type == 'T482_g4.7':
        T_TOA_sonora = 144.485 # K
        g = 501.187 # m/s^2
        R = Rj
        M = g*R**2/G
        filename = 'teff_482_grav_500_mh_+000_co_100_1e-8_df.pkl'
        df = pd.read_pickle(sonora_filepath + '/' + filename)
    elif atm_type == 'T2000_g5.0':        
        T_TOA_sonora = 717.83 # K
        g = 1000 # m/s^2
        R = Rj
        M = g*R**2/G
        filename = 'teff_2000_grav_1000_mh_+000_co_100_1e-8_df.pkl'
        df = pd.read_pickle(sonora_filepath + '/' + filename)  
    elif atm_type == 'T500_g5.0':
        T_TOA_sonora = 145.31 # K
        g = 1000 # m/s^2
        R = Rj
        M = g*R**2/G
        filename = atm_type+'_nc_moist_0.0metal_NR_smart.atm'
        df = pd.read_csv(sonora_filepath + '/' + atm_type + '_nc_moist_0.0metal_NR_smart.atm', sep='\s+', skiprows = 0)
    else:
        raise ValueError('atm_type must be one of implemented profiles.')
        
    # read in Sonora model data
    XH2 = np.array(df['H2']) # H2 mixing ratios
    P = np.array(df['pressure']) * 1e5 # pressure [bar], converted to [Pa]
    T = np.array(df['temperature']) # [K]
    mu = np.array(df['MU']) * 1.66054e-27 # [amu/molecule] converted to [kg/molecule]
    
    # interpolate
    P0 = 100000 # was 101325 Pa (1 atm), now 1 bar
    P_interp = np.sort(np.append(np.logspace(np.log10(P[0]), np.log10(P[-1]), 100000), P0)) # interpolate in log-log space for accuracy, making sure P0 is in the array
    T_interp = np.interp(P_interp, P, T)
    mu_interp = np.interp(P_interp, P, mu)
    XH2_interp = np.interp(P_interp, P, XH2)
    rho_interp = P_interp/(k*T_interp) * mu_interp # total mass density
    
    # construct Z(P)
    idx_P0 = np.where(P_interp==P0)[0][0] # P0 for integration constant / boundary condition
    integrand = T/(P*mu)  # function of P, w/ P0 = P(z=0km) = 1bar be definition
    spline_IntegrandofP = spline(np.log10(P), np.log10(integrand), k=1) # should be -integrand?
    integral = scipy.integrate.cumulative_trapezoid(10**spline_IntegrandofP(np.log10(P_interp)), P_interp, initial=0)
    Z0 = (k*integral/(G*M) + R**(-1))**(-1) - R
    Z = Z0 - Z0[idx_P0] # enforce constant of integration

    # construct H2 number density
    nH2_sonora_full = rho_interp * XH2_interp / mu_interp
    
    # extend isothermally (could also keep this as an analytic function instead of part of the interpolation)
    P_crit_Pa = (1e-6)*1e5 # 1e-6 bar is approximately where the Sonora model breaks down (transition to isothermal extension)
    z_crit = Z[np.where(P_interp < P_crit_Pa)[0][-1]]
    n_crit = nH2_sonora_full[np.where(P_interp < P_crit_Pa)[0][-1]]
    H0 = k*T_TOA_sonora/(mH2*g) # H at top of sonora
    z_isothermal = np.linspace(z_crit, z_max, 100)
    nH2_isothermal = n_crit*np.exp((R**2/H0) * ((z_isothermal + R)**(-1) - (z_crit+R)**(-1)))
    P_H2_isothermal = nH2_isothermal*T_TOA_sonora*k
    
    # total functions to interpolate
    z_sonora = Z[Z < z_crit]
    P_sonora = P_interp[Z < z_crit]
    nH2_sonora = nH2_sonora_full[Z < z_crit]
    P_H2_sonora = nH2_sonora_full[Z < z_crit]*T_interp[Z < z_crit]*k
    z_grid = np.hstack([z_sonora[::-1], z_isothermal])
    nH2_grid = np.hstack([nH2_sonora[::-1], nH2_isothermal])
    P_H2_grid = np.hstack([P_H2_sonora[::-1], P_H2_isothermal])
    
    # define "function" (spline interpolation) for ln(nH2(Z))
    ln_nH2 = spline(z_grid, np.log(nH2_grid), k=1)
    def get_nH2(z):
        return np.exp(ln_nH2(z))
    
    # define "function" (spline interpolation) for ln(P(Z))
    ln_PH2 = spline(z_grid, np.log(P_H2_grid), k=1)
    def get_PH2(z):
        return np.exp(ln_PH2(z))
    
    # define "function" (spline interpolation) for Z(ln(P))
    #get_z = spline(np.log(P_H2_grid), z_grid, k=1)
    
    return get_nH2, get_PH2


def construct_profile_Jupiter(atm_type, z_max, sonora_filepath):
    '''
    Read in Sonora model ouputs and construct spline interpolation functions for nH2(z),
    P(z), and z(P). Spline interpolation has been shown to be almost as fast, and
    more accurate, compared with fitting functional forms for these quantities.
    a polynomial fit function for these.
    For Jupiter, use Galileo data above the homopause rather than an isothermal
    approximation. 
    Inputs:
        z_max (float)): height at which electrons are removed from simulation [m]
        atm_type (str): which object to use
    Returns:
        get_ln_nH2 (function): spline interpolation of ln(nH2(z)) [m^-3]
        get_ln_P (function): spline interpolation of ln(P(z)) [Pa]
        get_Z (function): spline interpolation of z(ln(P)) [m]
    ''' 

    if atm_type == 'Jupiter':
        T_TOA_sonora = 150.78 # K
        g = 24.79 # m/s^2
        R = Rj
        M = g*R**2/G 
        filename = 'jupiter_1e-8_final.pkl'
        df = pd.read_pickle(sonora_filepath + '/' + filename)
    else:
        raise ValueError('This function is only applicable to Jupiter.')
        
    # read in Sonora model data
    XH2 = np.array(df['H2']) # H2 mixing ratios
    P = np.array(df['pressure']) * 1e5 # pressure [bar], converted to [Pa]
    T = np.array(df['temperature']) # [K]
    mu = np.array(df['MU']) * 1.66054e-27 # [amu/molecule] converted to [kg/molecule]
    
    # interpolate
    P0 = 100000 # was 101325 Pa (1 atm), now 1 bar
    P_interp = np.sort(np.append(np.logspace(np.log10(P[0]), np.log10(P[-1]), 100000), P0)) # interpolate in log-log space for accuracy, making sure P0 is in the array
    T_interp = np.interp(P_interp, P, T)
    mu_interp = np.interp(P_interp, P, mu)
    XH2_interp = np.interp(P_interp, P, XH2)
    rho_interp = P_interp/(k*T_interp) * mu_interp # total mass density
    
    # construct Z(P)
    idx_P0 = np.where(P_interp==P0)[0][0] # P0 for integration constant / boundary condition
    integrand = T/(P*mu)  # function of P, w/ P0 = P(z=0km) = 1bar be definition
    spline_IntegrandofP = spline(np.log10(P), np.log10(integrand), k=1) # should be -integrand?
    integral = scipy.integrate.cumulative_trapezoid(10**spline_IntegrandofP(np.log10(P_interp)), P_interp, initial=0)
    Z0 = (k*integral/(G*M) + R**(-1))**(-1) - R
    Z = Z0 - Z0[idx_P0] # enforce constant of integration

    # construct H2 number density
    nH2_sonora_full = rho_interp * XH2_interp / mu_interp
    
    # Seiff 1998 Galileo data
    chi_H2_Seiff = np.array([0.9828, 0.9866, 0.9886, 0.9886, 0.9846, 0.9716, 0.9300, 0.8890, 0.8673, 0.8621, 0.8620, 0.8620])
    mu_Seiff = np.array([2.001, 2.007, 2.013, 2.020, 2.034, 2.064, 2.151, 2.242, 2.296, 2.309, 2.309, 2.309]) * 1.66054e-27 # [kg]
    z_Seiff = np.array([1001, 900.6, 798.3, 699.6, 600.0, 500.0, 400.3, 350.3, 301.6, 201.0, 101.1, 22.67]) * 1e3 # [m]
    P_Seiff = np.array([0.00111, 0.00211, 0.00399, 0.00777, 0.0173, 0.0422, 0.143, 0.430, 2.02, 1.177e2, 6.703e3, 3.626e5]) * 0.1 # [Pa]
    T_Seiff = np.array([880.1, 863.5, 873.6, 743.2, 671.0, 548.2, 378.4, 208.5, 198.1, 157.4, 156.8, 120.9]) # [K]
    nH2_Seiff = P_Seiff*chi_H2_Seiff/(k*T_Seiff)  # P = nkT, n = number density
    
    # total functions to interpolate
    z_crit = 20e3
    z_sonora = Z[Z < z_crit]
    P_sonora = P_interp[Z < z_crit]
    nH2_sonora = nH2_sonora_full[Z < z_crit]
    P_H2_sonora = nH2_sonora_full[Z < z_crit]*T_interp[Z < z_crit]*k
    z_grid = np.hstack([z_sonora[::-1], z_Seiff[::-1]])
    nH2_grid = np.hstack([nH2_sonora[::-1], nH2_Seiff[::-1]])
    P_H2_grid = np.hstack([P_H2_sonora[::-1], P_Seiff[::-1]])
    
    # define "function" (spline interpolation) for ln(nH2(Z))
    ln_nH2 = spline(z_grid, np.log(nH2_grid), k=1)
    def get_nH2(z):
        return np.exp(ln_nH2(z))
    
    # define "function" (spline interpolation) for ln(P(Z))
    ln_PH2 = spline(z_grid, np.log(P_H2_grid), k=1)
    def get_PH2(z):
        return np.exp(ln_PH2(z))
    
    # define "function" (spline interpolation) for Z(ln(P))
    #get_z = spline(np.log(P_H2_grid), z_grid, k=1)
    
    return get_nH2, get_PH2

def moyal_mu(E_keV):
    '''
    Calculate fit moyal mu paramter for a given array of energies.
    Inputs:
        E_keV (numpy array): energy to calculate mu for [keV]
    Returns:
        mu (numpy array): mu paramter
    '''
    return -1.36601397e+00*np.log(E_keV) - 9.37779430e-02*np.log(E_keV)**2 + 6.83876054e-03*np.log(E_keV)**3 - 4.84590557e+01

def moyal_sigma(E_keV):
    '''
    Calculate fit moyal sigma paramter for a given array of energies.
    Inputs:
        E_keV (numpy array): energy to calculate sigma for [keV]
    Returns:
        sigma (numpy array): sigma paramter
    '''
    return -1.98630510e-02*np.log(E_keV) + 5.31896470e-03*np.log(E_keV)**2 - 4.48665267e-04*np.log(E_keV)**3 + 4.69517667e-01


def calc_q(z, E_keV, z_min, z_max, get_n_H2, event_type):
    '''
    Calculated parameterization curve for q for the given event. 
    Scalings (beta values in paper) are calculated in calculate_scalings_for_other_interactions.py (run_get_beta_scalings.sh)
    Inputs:
        z (numpy array): array of altitudes to calculate q over [m]
        E_keV (float): energy to calculate q for [keV]
        z_min (float): minimum altitude for calculating column density grid
        z_max (float): maxmimum altitude for calculating column density grid
        get_n_H2 (function): function for calculating number density of H2
        event_type (str): which event to calculate q for
    Returns:
        q (numpy array): event rate [number/m/incident electron]
    ''' 
    mu = moyal_mu(E_keV)
    sigma = moyal_sigma(E_keV)
    z_grid = np.linspace(z_min,z_max,10000) * u.m
    R_grid = construct_R_grid(z_grid.value, z_min, z_max, get_n_H2)
    N = get_column_density(z, z_grid.value, R_grid)/mH2 #* (u.m)**(-2) # #/m^2
    pdf = stats.moyal.pdf(-np.log(N), mu, sigma)*(N)**(-1) # pdf(N)
    nH2 = get_n_H2(z) #* (u.m**-3)
    Nevent_over_Ne = calc_Nevent_over_Ne(E_keV, event_type)
    q_event = pdf*nH2 * (Nevent_over_Ne)
    return q_event # * (u.m**-1) 

def calc_Nevent_over_Ne(E_keV, event_type):
    '''
    Calculated scaling for q for the given event. Beta_event from paper (ie. if ionizations, returns beta_ion)
    Inputs:
        E_keV (float): energy to calculate q for [keV]
        event_type (str): which event to calculate q for
    Returns:
        Nevent_over_Ne (float): ratio of total events to number incident electrons
    ''' 
    # A = scaling from ionization fit
    if event_type == 'Ionization heights [m]':
        A = 1
    elif event_type == 'Rotational excitation heights [m]':
        A = 22.732071222839828
    elif event_type == 'Vibrational excitation heights [m]':
        A = 3.8503848196709254
    elif event_type == 'Elastic scattering heights [m]':
        print('Elastic scattering is no longer tracked in the simulation.')
        return
    elif event_type == 'B excitation heights [m]':
        A = 0.36607970691253755
    elif event_type == 'C excitation heights [m]':
        A = 0.34638835897146375
    elif event_type == 'a excitation heights [m]':
        A = 0.013040842313136266
    elif event_type == 'b excitation heights [m]':
        A = 0.2747979462259085
    elif event_type == 'c excitation heights [m]':
        A = 0.020956886169008555
    elif event_type == 'e excitation heights [m]':
        A = 0.004118143179208906
    elif event_type == 'Themalization energy deposited [eV]': 
        # energy deposition due to thermalization, rot. and vib. excitation
        # NOTE: A has units here! 
        A = (2.2508978012636353e-05 * u.keV).to(u.eV).value # eV 
    elif event_type == 'Total energy deposition [eV]': 
        # energy deposition due to thermalization, rot. and vib. excitation
        # NOTE: A has units here! 
        #       A = beta_thermalization * E_thermalization + beta_rot * E_rot + beta_vib * E_vib
        #       beta as defined in Zuckerman et al. ~2025 paper
        E_rot_excitation = 0.0438 * u.eV 
        E_vib_excitation = 8.77596948e-01 * u.eV 
        A_thermalization = 2.2508978012636353e-05 * u.keV 
        A_rot = 22.732071222839828 * E_rot_excitation
        A_vib = 3.8503848196709254 * E_vib_excitation
        A = (A_thermalization + A_rot + A_vib).to(u.eV).value # eV
    else:
        raise ValueError('Event type must be one of applicable events.')
    a = 2.87448965
    b = 0.95527983
    return A * np.exp(a + b*np.log(E_keV))

def calc_Q(calc_Fe, zi, E_eV, z_min, z_max, get_n_H2, event_type):
    '''
    Calculated volumetric event rate (or energy deposition rate) for the given event. 
    Inputs:
        zi (int): altitude to calculate Q at [m]
        E_eV (numpy array of floats): energy to calculate Q over [eV] 
                            NB: expect eV, not keV like other functions!
        z_min (float): minimum altitude for calculating column density grid [m]
        z_max (float): maxmimum altitude for calculating column density grid [m]
        get_n_H2 (function): function for calculating number density of H2 [#/m^3]
        event_type (str): which event to calculate q for
        calc_Fe (function): function which returns the electron beam energy spectrum
                            to use, as as astropy quantity [# e-/cm^2/s/eV]
                            NB: returns cm^-2, not m like other functions!
    Returns:
        Q (numpy array): volumetric event rate [# events/cm^3/s]
                         or energy deposition rate [eV/cm^3/s] for event_type = 'total energy deposition'
    ''' 
    q = (calc_q(zi, E_eV/1e3, z_min, z_max, get_n_H2, event_type) * u.m**-1)
    F = calc_Fe(E_eV * u.eV) # (u.eV**-1 * u.s**-1 * u.m**-2)
    return trapezoid(q*F, E_eV * u.eV) # (u.cm**-3 * u.s**-1) # events/cm^3/s or eV/cm^3/s
