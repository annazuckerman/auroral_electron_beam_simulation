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

# define constants
c = 2.99e8 * u.m / u.s # speed of light [m/s]
c = c.to(u.m/u.s).value
m =  9.1093837e-31 * u.kg # e- mass [kg]
mH2 = 3.347649043E-27 # H2 mass [kg]
k = 1.380649e-23 # bolztman constant [J/k]
Rj = 71492e3 # Jupiter radius [m]
G = 6.6743e-11 # [Nm^2/kg^2]
e_ion = 15.43 # eV to ionize H2 . This is "binding energy" on NIST.
stop_height_pcntle = 0.01 # % of ionizations occuring below stopping height by definition
Rj = 71492e3 # Jupiter radius [m]

def n_H2_Jupiter_Hiraki(z):
    '''
    NOT USED
    NOTE: This has been shown to disagree both with the data from Seif 1998 and 
    with the results from the Sonora model.
    Return H2 number density at a given altitude for Jupiter. From Hiraki 2008. 
    Defined using z = 0km corresponds to P = 1 bar. 
    Inputs:
        z (numpy array or float): height(s) to determine altitude [m]
    Returns:
        n_H2 (numpy array or float:) number density of H2 at z [1/m^3]
    '''
    z1 = (300-z/1000)
    return 6.8e19*np.exp(z1/62) + 2.7e18*np.exp(z1/180) + 6e16*np.exp(z1/350)

def pick_theta(E, m, cdf_val):
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

def transform_collapsed(vx, vy, vz, dtheta):
    '''
    Return vector transformed into rotated frame, in cartesian coords
    Inputs:
        vx (numpy array ): x-components of vectors in old frame to rotate into new frame 
        vy (numpy array ): y-components of vectors in old frame to rotate into new frame
        vz (numpy array ): z-components of vectors in old frame to rotate into new frame
        dtheta (float): polar angle between old and new frames, measured
                        down from the old frame positive z axis to new frame 
                        positive z axis
    Returns:
        vx_prime (numpy array ): x-components of vectors in new frame 
        vy_prime (numpy array ): y-components of vectors in new frame         
        vz_prime (numpy array ): z-components of vectors in new frame 
    '''
    x_prime = vx*np.cos(dtheta) + vz*np.sin(dtheta)
    y_prime = vy
    z_prime = - vx*np.sin(dtheta) + vz*np.cos(dtheta)
    h_prime = np.sqrt(x_prime**2 + y_prime**2)
    return z_prime, h_prime


def save_state_min_store(E_now, cos_theta, dt, z_now, path, filename):
    '''
    Save the current simulation state to HDF.
    Inputs:
        E_now (numpy array): energies of electrons in simulation at current step [J]
        cos_theta (numpy array): cosine of angles of electron trajectories
        dt (numpy array): timstep value [s]
        z_now (numpy array): altitude [m]
        path (str): path to save file to
        filename (str): name of file
    Returns:
        None
    '''
    arr = list(zip(E_now, cos_theta, dt, z_now)) 
    df = pd.DataFrame(arr,  columns = ['E [J]', 'cos(theta)', 'dt [s]', 'z [m]'])
 
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_hdf(path + '/' + filename, key='state', mode='w')
    return

def save_results_min_store(z_ion, z_el_scat, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, path, filename):
    '''
    Save simulation results to HDF.
    Inputs:
        z_ion (numpy array): altitudes of ionizations [m]
        z_el_scat (numpy array): altitudes of elastic scatterings [m]
        z_rot_ex (numpy array): altitudes of rotational excitations [m]
        z_vib_ex (numpy array): altitudes of vibrational excitations [m]
        z_B_ex (numpy array): altitudes of B excitations [m]
        z_C_ex (numpy array): altitudes of C excitations [m]
        z_a_ex (numpy array): altitudes of a excitations [m]
        z_b_ex (numpy array): altitudes of b excitations [m]
        z_c_ex (numpy array): altitudes of c excitations [m]
        z_e_ex (numpy array): altitudes of e excitations [m]
        z_exit_z (numpy array): altitudes of exiting simulation due to altitude [m]
        z_exit_E (numpy array): altitudes of exiting simulation due to energy [m]
        E_exit_E (numpy_array): energies of electrons exiting due to energy [J]
        path (str): path to save file to
        filename (str): name of file
    Returns:
        None
    '''
    arr = [z_ion, z_el_scat, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E]
    df = pd.DataFrame(arr)
    df.index = ['Ionization heights [m]', 'Elastic scattering heights [m]', 'Rotational excitation heights [m]', 'Vibrational excitation heights [m]', 'B excitation heights [m]', 'C excitation heights [m]', 'a excitation heights [m]', 'b excitation heights [m]', 'c excitation heights [m]', 'e excitation heights [m]', 'Exit (altitude) heights [m]', 'Exit (energy) heights [m]', 'Exit (energy) energies [J]']

    if not os.path.exists(path):
        os.mkdir(path)
    df.to_hdf(path + '/' + filename, key='results', mode='w')
    return

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
    '''
    Generate an array of random numbers
    Inputs:
        N (int): number of random numbers to generate
    Returns:
        rand_nums (numpy array): array of random numbers (float) in [0,1]
        N_rand (int): number of random numbers
        counter (int): reset counter
    '''
    rand_nums = np.random.uniform(0,1,N)
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

def elastic_scat_xsec_old(E):
    '''
    Return elastic scattering cross section for H2 with incident e- at given energies.
    From fitting data from ____ using an _____.
    Inputs:
        E (numpy array ): energies of incident electrons (J)
    Returns:
        xsec (numpy array): cross section for H2 (m^2)
    '''
    x = E * 6.242e+18 # convert to eV
    # params from fitting 
    a = 28.3628867 
    b = -0.22793801
    c = 7.32464711
    d = 149.07169258
    y1 = 2*a/(1+np.exp(b*(x-c)))
    xsec = (2*d/y1)/(1 + 4*((x-c)/y1)**2) * (5.2918e-11)**2 # convert a0^2 to m^2   # * 10**(-16) / (1e4) # m^2
    return xsec

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

def get_pressure_Jupiter(z):   
    '''
    Map altitude (in m) to pressure level (in Pa) for Jupiter.
    Model from Sebastian Pineda, fit from data from Seif et. al. 1997.
    Defined using z = 0km corresponds to P = 1 bar. 
    Inputs:
        z (numpy array): altitude (m)
    Returns:
        P (numpy array): pressure (Pa)
    '''
    if (z < 0).any(): # Sonora gives P for lower altitudes
        raise ValueError('Altitude z=' + str(z) +  'm outside range for pressure profile.')
        
    Z = z/1000 # convert to km
    c1 = 4.90926165e-01
    c2 = 2.36511964e+01
    c3 = 2.42002670e-06
    c4 = 1.27021395e+02
    return (c1*np.exp(-Z/c2)  + c3*np.exp(-Z /c4)) * 1e5 # 1e5 converts to SI (Pa)

def get_pressure_T1400_g4_v1(z):   
    '''
    Map altitude (in m) to pressure level (in Pa) for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    TO DO: Though not neccessarily from a time efficiency standpoint (currenlty not used in the actual model), much prettier to
     fit a function to this, and would be faster.
    '''
    z = z/1000 # fitting was done with z in km, function takes z in m
    # from fitting calculated z(P) using data from SONORA model
    A =  -0.019800107761776466
    B =  -2.943539832233964e-05
    C =  -1.8325901230837773e-08
    D =  1.2157499523060282e-12
    E =  11.424003311238438
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E)

def get_pressure_T1400_g4(z):
    '''
    Map altitude (in m) to pressure level (in Pa) for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -800e3).any() | (z > 500e3).any():
        #raise ValueError('Altitude z=' + str(z) +  'm outside range for pressure profile.')
        print('WARNING: Altitude z=' + str(z[(z > 500e3) | (z < -800e3)]) +  'm outside range for pressure profile.')
        
    A =  -1.8590892442668644e-05
    B =  -3.694890367693835e-11
    C =  -4.114357032511582e-17
    D =  6.727399542262329e-23
    E =  1.2660219341204418e-28
    F =  -2.6929904945925236e-34
    G =  -4.880850385117377e-40
    H =  4.073237856682065e-46
    I =  1.1100849110916402e-51
    J =  5.321192764810568e-58
    K =  11.500607161096138
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)

def get_pressure_T1400_g5(z):
    '''
    Map altitude (in m) to pressure level (in Pa) for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -65e3).any() | (z > 44e3).any():
        #raise ValueError('Altitude z=' + str(z) +  'm outside range for pressure profile.')
        print('WARNING: Altitude z=' + str(z[(z > 44e3) | (z < -65e3)]) +  'm outside range for pressure profile.')
        
    A =  -0.00024187703016159077
    B =  -4.5888004911959196e-09
    C =  -3.723397243075698e-14
    D =  4.0429702934977004e-19
    E =  1.0643374977980512e-24
    F =  2.006316375144264e-28
    G =  1.1791894781586194e-32
    H =  1.531970584062757e-38
    I =  -3.637029997861661e-42
    J =  -3.2087039649607437e-47
    K =  11.503167790469629
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)

def get_pressure_T900_g4(z):
    '''
    Map altitude (in m) to pressure level (in Pa) for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -660e3).any() | (z > 257e3).any():
        #raise ValueError('Altitude z=' + str(z) +  'm outside range for pressure profile.')
        print('WARNING: Altitude z=' + str(z[(z > 257e3) | (z < -660e3)]) +  'm outside range for pressure profile.')
        
    A =  -2.7670989797036807e-05
    B =  -1.0484504900316648e-10
    C =  -3.5441861626889795e-16
    D =  -4.039406249751632e-22
    E =  1.5047974877688946e-27
    F =  5.4418928786182124e-33
    G =  4.2476945194146285e-39
    H =  -5.837127046807449e-45
    I =  -1.1393975721037803e-50
    J =  -5.174230785504113e-57
    K =  11.518882268443297
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)

def get_pressure_T900_g5_v1(z):   
    '''
    Map altitude (in km) to pressure level (in bar) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    TO DO: Though not neccessarily from a time efficiency standpoint (currenlty not used in the actual model), much prettier to
     fit a function to this, and would be faster.
    '''
    z = z/1000 # fitting was done with z in km, function takes z in m
    
    # from fitting calculated z(P) using data from SONORA model
    A =  -0.41228825667941105
    B =  -0.012422694360803836
    C =  -0.00014436097354687932
    D =  2.2012527415766286e-07
    E =  11.419393464388618
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E)

def get_pressure_T900_g5(z):
    '''
    Map altitude (in km) to pressure level (in bar) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -50e3).any() | (z > 25e3).any():
        #raise ValueError('Altitude z=' + str(z) +  'm outside range for pressure profile.')
        print('WARNING: Altitude z=' + str(z[(z > 25e3) | (z < -50e3)]) +  'm outside range for pressure profile.')
        
    A =  -0.00037299128091289893
    B =  -1.4554536433882855e-08
    C =  -4.426280878593849e-13
    D =  -5.788202814658167e-19
    E =  4.976162157898435e-22
    F =  1.4229523038059653e-26
    G =  -4.7734870517547344e-32
    H =  -9.880343953874867e-36
    I =  -1.938117246538918e-40
    J =  -1.2674582618021346e-45
    K =  11.499677903158654
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)

def get_pressure_T482_g4point7(z):
    '''
    Map altitude (in km) to pressure level (in bar) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -68.82e3).any() | (z > 23.44e3).any():
        #raise ValueError('Altitude z=' + str(z) + 'm outside range for density profile.')
        print('WARNING: Altitude z=' + str(z[(z > 23.44e3) | (z < -68.82e3)]) +  'm outside range for pressure profile.')
        
    A =  -0.0003097214634231771
    B =  -1.7232364406652063e-08
    C =  -7.962307070670462e-13
    D =  -2.635144208577491e-19
    E =  1.2355382698211665e-21
    F =  2.738707036511675e-26
    G =  -4.460248606976265e-31
    H =  -2.3943545838589527e-35
    I =  -3.1703776287783155e-40
    J =  -1.4186721310940552e-45
    K =  11.577958032426833
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)

def get_pressure_T2000_g5(z):
    '''
    Map altitude (in km) to pressure level (in bar) for T2000g5.0 planet.
    From SONORA model. 
    Inputs:
     z (numpy array): altitude (m)
    Returns:
     P (numpy array): pressure (Pa)
    '''
    if (z < -86.535e3).any() | (z > 56.644e3).any():
        #raise ValueError('Altitude z=' + str(z) + 'm outside range for density profile.')
        print('WARNING: Altitude z=' + str(z[(z > 56.644e3) | (z < -86.535e3)]) +  'm outside range for pressure profile.')
        
    A =  -0.00017064408528387782
    B =  -2.766979271180578e-09
    C =  -2.337441407779075e-14
    D =  2.2455220671820514e-19
    E =  2.3973595331398726e-24
    F =  -9.290510019093494e-31
    G =  1.0540624909974553e-33
    H =  5.708180241559639e-39
    I =  -2.1151564243275053e-43
    J =  -1.7018309254650682e-48
    K =  11.477740455450759
    return np.exp(A*z + B*z**2 + C*z**3 + D*z**4 + E*z**5 + F*z**6 + G*z**7 + H*z**8 + I*z**9 + J*z**10 + K)


def get_z_T900_g5_v1(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    A =  0.47374316386539633
    B =  -0.9313338437460462
    C =  0.1747847749317132
    D =  0.2773043948354066
    E =  17.257430416452117
    return (-C* np.exp(D*np.log(P)+A) + B*np.log(P) + E) / 1000 # fitting was done with z in km, function takes z in m

def get_z_T900_g5(P):
    '''
    Map altitude (in Pa) to pressure level (in m) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -1008.4052406702094
    B =  -4.260099454903526
    C =  -1.8525949064760352
    D =  -0.2711427551112937
    E =  0.013697782563635399
    F =  0.0044184170522237045
    G =  -0.0002554494233445243
    H =  -4.0725309840405754e-05
    I =  3.5376106946694257e-06
    J =  -7.612028130419894e-08
    K =  16646.6866791246
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)

def get_z_T900_g4(P):
    '''
    Map altitude (in Pa) to pressure level (in m) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -9895.79183490076
    B =  -162.42467757950772
    C =  -36.441832622872596
    D =  -1.561298953557048
    E =  0.6335692958661515
    F =  0.005498373319603014
    G =  -0.0071769197601294216
    H =  0.0001650655530378742
    I =  1.2176960066312122e-05
    J =  -3.9209188674234113e-07
    K =  191772.91335624774
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)

def get_z_T1400_g5(P):
    '''
    Map altitude (in Pa) to pressure level (in m) for T900g5.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -1964.6808289638616
    B =  2.2592755433088962
    C =  -4.591848842375531
    D =  -0.3673286492629777
    E =  0.06975012409390112
    F =  -0.00042868579127543975
    G =  -0.0005597870527811892
    H =  4.371568786595226e-05
    I =  -1.9933025698745216e-06
    J =  4.240496013573638e-08
    K =  29557.74996760181
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)

def get_z_T1400_g4_v1(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    A =  1.6607288596279361
    B =  -18.20338544620369
    C =  1.7046215128184972
    D =  0.24980444046806255
    E =  359.50007656323123
    return (-C* np.exp(D*np.log(P)+A) + B*np.log(P) + E) / 1000 # fitting was done with z in km, function takes z in m

def get_z_T1400_g4(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for T1400g4.0 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -20464.00785938137
    B =  -276.958721111121
    C =  -44.96855806135498
    D =  -4.862304610485959
    E =  0.48197977600240305
    F =  0.12547732763019906
    G =  -0.009728228380863457
    H =  -0.0010920105010596515
    I =  0.00011010320264689915
    J =  -2.4977004051789255e-06
    K =  344934.4068324528
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)

def get_z_T482_g4point7(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for T482g4.7 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -1028.2509267122662
    B =  10.389193511691415
    C =  2.5425096052958303
    D =  -0.6354249005619377
    E =  -0.09023595080743538
    F =  0.015519387685578223
    G =  0.00026352298325329885
    H =  -0.00016048214220796596
    I =  9.140471003938138e-06
    J =  -1.6045261463407726e-07
    K =  16379.176972480527
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)

def get_z_T2000_g5(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for T2000g5 planet.
    From SONORA model. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    ''' 
    if (P < -1e-1).any() | (P > 1e8).any():
        raise ValueError('Pressure P=' + str(P) +  'm outside range for z(P) profile.')
        
    A =  -2530.5186664427497
    B =  6.273524001347053
    C =  -2.910632591726845
    D =  -1.3823320708299496
    E =  0.012093460495166169
    F =  0.02706439792127364
    G =  -0.0011075506005526305
    H =  -0.00019923303630324664
    I =  1.5911177924871367e-05
    J =  -3.188359530483552e-07
    K =  39139.28902782731
    x = np.log(P)
    return (A*x + B*x**2 + C*x**3 + D*x**4 + E*x**5 + F*x**6 + G*x**7 + H*x**8 + I*x**9 + J*x**10 + K)


def get_z_Jupiter(P):
    '''
    Map pressure level (in m) to altitude (in Pa) to for Jupiter.
    From Fitting Seiff 1997 data. 
    Inputs:
     P (numpy array): pressure (Pa)
    Returns:
     z (numpy array): altitude (m)
    TO DO: find a function that isn't piecewise for computational efficiency
    ''' 
    if (P > 100000).any(): # Sonora gives P for lower altitudes
        raise ValueError('Pressure P=' + str(P) +  'm outside range for pressure profile.')
        
    A =  -126866.10987817039
    B =  -177391.6086945308
    C =  -23835.464567413946
    D =  259217.65481700914
    logP = np.log(P)
    t = -4 
    return (A*logP + B) * np.heaviside(t - logP, 0) + (C*logP + D) * np.heaviside(logP - t, 0)

def get_stopping_height(path, run_ids):
    '''
    Get stopping height from run results. 
    Inputs:
     path (str): path to results files
    Returns:
     run_ids (list of strings): run IDs
    ''' 
    stopping_heights = []
    for run_id in run_ids:
        results = pd.read_csv(path + '/results_' + run_id + '.csv', index_col=0, nrows = 1)
        z = results.loc['Ionization heights [m]'].dropna()
        stopping_heights += [np.percentile(z, stop_height_pcntle)] # 99% of ionizations occur above this height
    return stopping_heights

def get_uncertainties(z_arr, nfolds, bins, Ntot, stop_height_pcntle, get_n_H2, get_pressure, z_max, atm_type, min_val, max_val, make_plot = False, make_subset_plot = False):
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
        # z_remaining = np.array(list(set(list(z_remaining)) - set(list(fold))))
        
        # collect bin values
        counts, bins = np.histogram(fold, bins = bins)
        all_folds_binned[i,:] = counts
        #counts = counts / Ntot # dividing here just divides the resulting SD
        
        # collect R0 values
        stopping_height = np.percentile(fold, stop_height_pcntle)
        stopping_P = get_pressure(stopping_height)
        all_stopping_P[i] = stopping_P
        N0, err = quad(get_n_H2, stopping_height, z_max)
        R0 = N0*mH2 # column density at stopping height
        all_R0[i] = R0
        
        # collect z-peak values
       # print('fold:',fold)
       # print('bins:', bins)
        z_peak = get_kde_values(fold, bins, Ntot, atm_type, get_n_H2, min_val, max_val, make_plot = False)[1]
        P_peak = get_pressure(z_peak)
        z_grid = np.linspace(0,max_val,10000)
        R_grid = construct_R_grid(z_grid, min_val, max_val, get_n_H2)
        R_peak = get_column_density(z_peak, z_grid, R_grid) #quad(get_n_H2, z_peak, z_max)[0]*mH2
        all_R_peaks[i] = R_peak/R0
        all_RoverR0_peaks[i] = R_peak/R0
       # print('z_peak:', z_peak)
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
   # sd_tot = N/N_per_fold * sd
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
   # npoints = 100
    if return_kde_curve_median_and_sd:
        all_kde_curve_values = np.zeros([nfolds, npoints])*np.nan
    for i in range(nfolds):
        #np.random.seed(42)
        fold = np.random.choice(events_remaining, N_per_fold, replace = True) # choose with replacement
        # z_remaining = np.array(list(set(list(z_remaining)) - set(list(fold))))
        
        # collect bin values
        counts, bins = np.histogram(fold, bins = bins)
        all_folds_binned[i,:] = counts
        #counts = counts / Ntot # dividing here just divides the resulting SD
        
        # collect z-peak values
       # print('fold:',fold)
       # print('bins:', bins)
        if return_kde_curve_median_and_sd:
            peak_val, peak_loc, kde_curve = get_kde_values(fold, bins, Ntot, atm_type, get_nH2, min_val, max_val, npoints, make_plot = False, return_kde_curve = True)
        else:
            peak_val, peak_loc = get_kde_values(fold, bins, Ntot, atm_type, get_nH2, min_val, max_val, npoints, make_plot = False, return_kde_curve = False)
        all_peak_locs[i] = peak_loc
        all_peak_vals[i] = peak_val
        if return_kde_curve_median_and_sd:
            all_kde_curve_values[i,:] = kde_curve
        
        if make_subset_plot:
            #print('making subset histogram plot')
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
  
   # sd_tot = N/N_per_fold * sd
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
        fig1.savefig('./test_plots/kde_histograms_test6.png')#, dpi = 300)
        plt.close(fig1)

    if return_kde_curve:
         return peak_val, peak_loc, stats_kde_curve
    else:
        return peak_val, peak_loc #, R0   
        
def get_Hiraki_parameterization_curve(z, z_min, z_max, e0, get_nH2):#, stopping_height):
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

def calculate_derived_quantities(file_path, nbins, plot_type, atm_type, Ntot, energy, indep_var, quantity): 
    '''
    Calculate (without plotting) histogram of the given quantity as a function of the given independent variable, 
    for hte specified energy and interaction.
    Inputs:
        file_path (str): filepath for reading in results data
        nbins (int): number of bins for histogram
        int_type (int): type of event to histogram (corresponds
                        to possible interaction codes)
        atm_type (str): which profile to use for conveting z to P
        energy (str): energy to use, in format [x.xkeV, y.ykeV]
        Ntot (int): total number primary electrons
        indep_var (string): what to calculate the histogram over ('PandZ' or 'RoverR0')
        quantity (string): what to plot ('q' = normalized frequency, 'lambda' = e0*q*R0/e_ion*rho)
    Returns:
        binned_quantity (numpy array of floats): the histogramed quantity
        bin_edges (numpy array of floats): histogram bins
    '''
     
    if (-2 == plot_type):
        row = 'Exit (energy) heights [m]'
    elif (-1 == plot_type):
        row = 'Ionization heights [m]'
    elif (1 == plot_type): 
        row = 'Ionization heights [m]'
    elif (2 == plot_type):
        row = 'Rotational excitation heights [m]'
    elif (3 == plot_type):
        row = 'Elastic scattering heights [m]'
    elif (4 == plot_type):
        row = 'B excitation heights [m]'
    elif (5 == plot_type):
        row = 'C excitation heights [m]'
    elif (6 == plot_type):
        row = 'a excitation heights [m]'
    elif (7 == plot_type):
        row = 'b excitation heights [m]'
    elif (8 == plot_type):
        row = 'c excitation heights [m]'
    elif (9 == plot_type):
        row = 'e excitation heights [m]'
    elif (10 == plot_type):
        row = ['Vibrational excitation heights [m]']
    else: raise ValueError('Plot type must be allowed interaction code.')
         
    if atm_type == 'Jupiter':
        z_max = 2000e3
        z_min = 100e3
        plot_xmin = 2e-8
        plot_zmin = 100e3
        RoverR0_min = 1e-6
        RoverR0_max = 5e0
        get_n_H2 = n_H2_Jupiter
        get_pressure = get_pressure_Jupiter
    elif atm_type == 'T1400_g4.0':
        z_max = 486.71e3
        z_min = 250e3
        plot_zmin = 250e3
        plot_xmin = 1e-7
        RoverR0_min = 1e-6
        RoverR0_max = 5e0
        get_n_H2 = n_H2_T1400_g4
        get_pressure = get_pressure_T1400_g4
    elif atm_type == 'T900_g5.0':
        z_max = 23.75e3
        z_min = 11.5e3
        plot_zmin = 11.5e3
        plot_xmin = 2e-6
        RoverR0_min = 1e-6
        RoverR0_max = 5e0
        get_n_H2 = n_H2_T900_g5
        get_pressure = get_pressure_T900_g5
    else:
         raise ValueError('atm_type must be one of implemented profiles.')
    
    # construct grid of mass column density for interplation
    z_grid = np.linspace(0,z_max,100)
    R_grid = construct_R_grid(z_grid)
   
    e0 = float(energy.split('k')[0]) * 1e3 # eV
    results_file = [file for file in os.listdir(file_path + '/' + energy) if file.startswith('results')][0]
    df = pd.read_hdf(file_path + '/' + energy + '/' + results_file, 'results') 
    z_ions = df.loc['Ionization heights [m]'].dropna()
    stopping_height = np.percentile(z_ions, stop_height_pcntle)
    N0, err = quad(get_n_H2, stopping_height, z_max)
    R0 = N0*mH2 # column density at stopping height
    z_events = df.loc[row].dropna()
    R_events = get_column_density(z_events, z_grid, R_grid) # interpolating is faster for thousands of events
    RoverR0_events = R_events/R0
    if (indep_var == 'PandZ') or (indep_var == 'P') or (indep_var == 'Z'):
        bins_arr = np.linspace(z_min, z_max, nbins)
        binwidth = bins_arr[1] - bins_arr[0]
        counts, bins = np.histogram(z_events, bins = bins_arr) # = np.histogram(df.loc[rows[j]], bins = nbins)#, density = norm) 
    elif indep_var == 'RoverR0':
        bins_arr = np.logspace(np.log10(RoverR0_min), np.log10(RoverR0_max), nbins) #np.linspace(RoverR0_min, RoverR0_max, nbins[k])
        if quantity == 'lambda':
            bins_arr = np.linspace(0,1,nbins)
        counts, bins = np.histogram(RoverR0_events, bins = bins_arr)
    counts = counts / Ntot
    binwidth = (bins[1]-bins[0])
    bincenters = 0.5*(bins[1:]+bins[:-1]) - binwidth  
    q = counts/binwidth # this is raw bin counts/ Ntot / binwidth
    if quantity == 'q':
        hist_values = q
        if indep_var == 'Z':
            nfolds = 100
            bins_sd, z_peak_sd, z_peak_mean, P_peak_sd, P_peak_mean, RoverR0_peak_sd, RoverR0_peak_mean, R_peak_sd, R_peak_mean, R0_sd, R0_mean, stopping_P_sd, stopping_P_mean = get_uncertainties(z_events, nfolds, bins_arr, Ntot, stop_height_pcntle, get_n_H2, get_pressure, z_max, atm_type)
            bins_err = 3*bins_sd/binwidth # use 3 sigma error bars 
    elif indep_var == 'RoverR0':
        bins_z =  get_z_from_column_density(bins*R0, z_grid, R_grid)[::-1] # convert R bins to z for normalizing by binwidth
        binwidths_z = bins_z[1:] - bins_z[:-1]
        q = counts/binwidths_z[::-1]  # this is raw counts / Ntot / binwidth [m]
        if quantity == 'lambda':
            rho = get_n_H2(bins_z[-2::-1]) * mH2 # get_n_H2(bins_z[:-1]) * mH2
            lam = (e_ion/e0) * (q * R0 / rho)
            hist_values = lam          
    return bins_arr, hist_values

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
        get_Z (function): spline interpolation of z(ln(P)) [m]
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
    z_isothermal = np.linspace(z_crit, z_max, 100)  # zmax should be calculated as ((H0/R**2)*np.log(P_max/P0) + (z0+R)**(-1))**(-1) - R but must have a typo
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
    
    return get_nH2, get_PH2

def calc_q(z, E_keV, z_min, z_max, get_n_H2, event_type):
    '''
    Calculated parameterization curve for q for the given event. 
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
    # NB: mu and sigma reported in paper (Zuckerman ~2025) are actually mu' = mu*ln10, sigma' = sigma*ln10
    #     (ie. the values of mu and sigma reported here are for the Moyal distribution for -log10(N), 
    #     whereas those reported in the paper text are for the Moyal distribution for -ln(N), which
    #     was done for clarity in the paper, and means multiplying the values here by ln10)
    mu =  -0.59175*np.log(E_keV) + -0.031*np.log(E_keV)**2 + 0.004*np.log(E_keV)**3 + -21.06433
    sigma = -0.00563*np.log(E_keV) + 0.00375*np.log(E_keV)**2 + -0.00053*np.log(E_keV)**3 + 0.20097
    z_grid = np.linspace(z_min,z_max,10000) * u.m
    R_grid = construct_R_grid(z_grid.value, z_min, z_max, get_n_H2)
    N = get_column_density(z, z_grid.value, R_grid)/mH2 #* (u.m)**(-2) # #/m^2
    pdf = stats.moyal.pdf(-np.log10(N), mu, sigma)*(N*np.log(10))**(-1) # pdf(N)
    nH2 = get_n_H2(z) #* (u.m**-3)
    Nevent_over_Ne = calc_Nevent_over_Ne(E_keV, event_type)
    q_ion = pdf*nH2 * (Nevent_over_Ne)
    return q_ion # * (u.m**-1) 

def calc_Nevent_over_Ne(E_keV, event_type):
    '''
    Calculated parameterization curve for q for the given event. 
    Inputs:
        E_keV (float): energy to calculate q for [keV]
        event_type (str): which event to calculate q for
    Returns:
        Nevent_over_Ne (float): ratio of total events to number incident electrons
    ''' 
    # A = scaling from ionization fit
    if event_type == 'Ionization heights [m]':
        A = 0
    elif event_type == 'Exit (energy) heights [m]':
        A = -0.71441381
    elif event_type == 'Rotational excitation heights [m]':
        A = 1.87891836
    elif event_type == 'Vibrational excitation heights [m]':
        A = 0.72903493
    elif event_type == 'Elastic scattering heights [m]':
        A = 4.64508072
    elif event_type == 'B excitation heights [m]':
        A = -1.01756976
    elif event_type == 'C excitation heights [m]':
        A = -1.07435311
    elif event_type == 'a excitation heights [m]':
        A = -3.43820693
    elif event_type == 'b excitation heights [m]':
        A = -1.29520409
    elif event_type == 'c excitation heights [m]':
        A = -3.26708364
    elif event_type == 'e excitation heights [m]':
        A = -4.97565391
    elif event_type == 'total energy deposition': 
        # energy deposition due to thermalization, rot. and vib. excitation
        # NOTE: A has units here! 
        #       exp(A) = beta_thermalization * E_thermalization + beta_rot * E_rot + beta_vib * E_vib
        #              = 6.54*4.38e-2 + 2.07*8.77e-2 + 4.9e-1*0.6
        #       beta as defined in Zuckerman et al. ~2025 paper
        A = -0.27180 # eV 
    else:
        raise ValueError('Event type must be one of applicable events.')
    a = 0.913336
    b = -0.05016688
    c = -0.00357498
    d = 3.02340139
    logE = np.log(E_keV)
    return np.exp(A + (a*logE + + b*logE**2 + c*logE**3 + d))

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
                            NB: returns cm^-3, not m like other functions!
    Returns:
        Q (numpy array): volumetric event rate [# events/cm^3/s]
                         or energy deposition rate [eV/cm^3/s] for event_type = 'total energy deposition'
    ''' 
    q = (calc_q(zi, E_eV/1e3, z_min, z_max, get_n_H2, event_type) * u.m**-1)
    F = calc_Fe(E_eV * u.eV) # (u.eV**-1 * u.s**-1 * u.m**-2)
    return trapezoid(q*F, E_eV * u.eV) # (u.cm**-3 * u.s**-1) # e-/cm^3/s
