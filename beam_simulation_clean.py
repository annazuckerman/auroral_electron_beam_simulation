version = 'v23'

# Interaction codes
# '-3' --> particle exit simulation due to reaching max height
# '-2' --> particle exit main simulation loop due to low energy
# '-1' --> particle creation
# '0'  --> no interaction
# '1'  --> ionization
# '2'  --> rotational excitation 
# '3'  --> elastic scattering
# '4'  --> B excitation
# '5'  --> C excitation
# '6'  --> a excitation
# '7'  --> b excitation
# '8'  --> c excitation
# '9'  --> d excitation
# '10' --> vibrational excitation
# NOTE: an "interaction" will have a code > 0

# imports
import os
import utils
import numpy as np
import pandas as pd
from astropy import units as u
import time as tm
from scipy import integrate
import datetime
import argparse
from utils import ionization_xsec 
from utils import elastic_scat_xsec
from utils import B_excitation_xsec
from utils import C_excitation_xsec
from utils import a_excitation_xsec
from utils import b_excitation_xsec
from utils import c_excitation_xsec
from utils import e_excitation_xsec
from utils import vib_excitation_xsec
import traceback
t = datetime.datetime.now()
t = t.strftime('%m.%d.%Y.%H.%M')

# parse input args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--Ne0',
                    dest='Ne0',
                    type=int,
                    help='Number of incident electrons to simulate.',
                    required=True)
parser.add_argument('--e0',
                    dest='e0',
                    type=float,
                    help='Incident electron beam energy in keV.',
                    required=True)
parser.add_argument('--cos_theta',
                    dest='cos_theta',
                    type=float,
                    help='Consine of incident electron beam angle.',
                    required=True)
parser.add_argument('--vary_theta',
                    dest='vary_theta',
                    action='store_true',
                    help='Whether we are testing a range of theta values in this set of simulations.',
                    default=False)
parser.add_argument('--atm_type',
                    dest='atm_type',
                    type=str,
                    help='Which atmospheric profile to use.',
                    required=True)
parser.add_argument('--picaso_filepath',
                    dest='picaso_filepath',
                    type=str,
                    help='Relative path to Picaso atmospheric profiles.',
                    required=True)
parser.add_argument('--d',
                    dest='diagnostics',
                    action='store_true',
                    help='Whether to output some extra diagnostic data.',
                    default=False)
parser.add_argument('--s',
                    dest='save_state',
                    action='store_true',
                    help='Whether save the final simulation state.',
                    default=False)
parser.add_argument('--min',
                    dest='minimum_store',
                    action='store_true',
                    help='Whether store only minimal run results, for speed.',
                    default=False)
parser.add_argument('--r',
                    dest='restore_state',
                    action='store_true',
                    help='Whether we are re-starting from a restored state.',
                    default=False)
parser.add_argument('--r_filepath',
                    dest='restore_filepath',
                    type=str,
                    help='If starting from restored state, path to state files.')
parser.add_argument('--r_run_ID',
                    dest='restore_run_ID',
                    type=str,
                    help='If starting from restored state, ID of run to restart.')
parser.add_argument('--logfile',
                    dest='logfile',
                    type=str,
                    help='Name of run logfile.')
parser.add_argument('--Jup_isothermal',
                    dest='Jup_isothermal',
                    action='store_true',
                    help='Whether to run Jupiter using an isothermally extended profile.',
                    default=False)
parser.add_argument('--Jup_H2008',
                    dest='Jup_H2008',
                    action='store_true',
                    help='Whether to run Jupiter using the density profile from Hiraki + Tao 2008.',
                    default=False)
parser.add_argument('--exit_pcnt',
                    dest='exit_pcnt',
                    type=float,
                    help='Fraction of electrons below 1eV threshold to end simulation.',
                    default=0.99)

args = parser.parse_args()
Ne = args.Ne0
e0 = args.e0
e0 = e0 * u.keV
cos_theta_0 = args.cos_theta
atm_type = args.atm_type
diagnostics = args.diagnostics
save_state = args.save_state
restore_state = args.restore_state
minimum_store = args.minimum_store
vary_theta = args.vary_theta
logfile = args.logfile
picaso_filepath = args.picaso_filepath
Jup_isothermal = args.Jup_isothermal 
Jup_H2008 = args.Jup_H2008
exit_pcnt = args.exit_pcnt

# define inputs
m =  9.1093837e-31 * u.kg # e- mass [kg]
mH2 = 3.347649043E-27 * u.kg # H2 mass [kg]
c = 2.99e8 * u.m / u.s # speed of light [m/s]
k = 1.380649e-23 * u.Joule / u.K # bolztman constant [J/k]
E_rot_excitation = 0.0438 * u.eV # 50 * u.eV # [eV] energy lost to excitation in excitation interaction, from MCCC database "threshold energy"
E_vib_excitation = 8.77596948e-01 * u.eV # "threshold energy" from MCCC database
E_C_excitation = 1.22910e1 * u.eV # "threshold energy" from MCCC database
E_B_excitation = 1.11829e1 * u.eV # "threshold energy" from MCCC database
E_a_excitation = 1.17934e1 * u.eV # "threshold energy" from MCCC database
E_b_excitation = 4.47713e0 * u.eV # "threshold energy" from MCCC database
E_c_excitation = 1.21569e1 * u.eV # "threshold energy" from MCCC database
E_e_excitation = 1.32260E+01 * u.eV # "threshold energy" from MCCC database
E_threshold = E_b_excitation # 1.60218e-19 # 1 eV in J, energy for which ionization cross section equation doesn't hold
E_ion = 15.43 * u.eV # eV to ionize H2. This is "binding energy" on NIST
    
if Jup_isothermal:
    print()
    print('WARNING: USING ISOTHERMALLY EXTENDED DENSITY PROFILE FOR JUPITER.')
    print()
       
# pick which density profile to use
if atm_type == 'Jupiter': 
    g = 24.79 * u.m/u.s**2 # NASA Jupiter fact sheet
    H0 = 2000e3 # [m]  2000 "top" of atmosphere altitude [km], P ~ 10^-13 bars
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 150.78 # [K] T at top of atm from Picaso, 10^-8 bars
    from utils import rot_excitation_xsec_Jupiter as rot_excitation_xsec
    if Jup_isothermal: # To use isothermal extension above Picaso model
        get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
    elif Jup_H2008: # To use profile from Hiraki and Tao 2008
        get_n_H2 = utils.n_H2_Jupiter_Hiraki 
    else: # using Galileo data by default
        get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, max_height, picaso_filepath) 
    if uniform_atm_test_run:
        H0 = 0
        get_n_H2, get_P_H2 = utils.construct_fake_uniform_profile(atm_type, max_height, picaso_filepath)        
elif atm_type == 'T1400_g4.0':
    g = 100 * u.m/u.s**2  # log g = 4.0 in cgs 
    H0 = 755.560e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 550.72 # [K] T at top of atm from Picaso, 10^-8 bars
    from utils import rot_excitation_xsec_1400K as rot_excitation_xsec
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T1400_g5.0':
    from utils import rot_excitation_xsec_1400K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2  # log g = 4.0 in cgs 
    H0 =  71.970e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 575.509596 # [K] T at top of atm from Picaso, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T900_g5.0':
    H0 = 37.612e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs
    T = 283.47 # [K] T at top of atm from Picaso, 10^-8 bars
    from utils import rot_excitation_xsec_900K as rot_excitation_xsec
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T900_g4.0':
    from utils import rot_excitation_xsec_900K as rot_excitation_xsec
    g = 100 * u.m/u.s**2 # log g = 5.0 in cgs 
    H0 = 398.090e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 263.422216 # [K] T at top of atm from Picaso, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T482_g4.7':
    from utils import rot_excitation_xsec_482K as rot_excitation_xsec
    g = 501.187 * u.m/u.s**2 # log g = 5.0 in cgs 
    H0 = 37.942e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 144.485 # [K] T at top of atm from Picaso, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T2000_g5.0':
    from utils import rot_excitation_xsec_2000K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs 
    H0 = 92.236e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 717.83 # [K] T at top of atm from Picaso, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)
elif atm_type == 'T500_g5.0':
    from utils import rot_excitation_xsec_500K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs
    H0 = 19.4816e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 145.31 # [K] T at top of atm from Picaso, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, picaso_filepath)    
else:
    print('Specified density profile is not implemented.')


# convert everything to standard SI units
e0_kev = e0.to(u.keV).value
e0 = e0.to(u.J).value
m = m.to(u.kg).value
mH2 = mH2.to(u.kg).value
E_rot_excitation = E_rot_excitation.to(u.J).value
E_vib_excitation = E_vib_excitation.to(u.J).value
E_B_excitation = E_B_excitation.to(u.J).value
E_C_excitation = E_C_excitation.to(u.J).value
E_a_excitation = E_a_excitation.to(u.J).value
E_b_excitation = E_b_excitation.to(u.J).value
E_c_excitation = E_c_excitation.to(u.J).value
E_e_excitation = E_e_excitation.to(u.J).value
E_ion = E_ion.to(u.J).value
c = c.to(u.m/u.s).value
g = g.to(u.m/u.s**2).value
k = k.to(u.Joule/u.K).value

# read in restored state if restoring from previous run
if restore_state:
    restore_filepath = args.restore_filepath
    restore_run_ID = args.restore_run_ID
    E_now, cos_theta, dt, ints, z_now, v_z_now, v_h_now, Ne, Ne_tot, z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, E_thermalization, z_thermalization, i, N_oos_E, N_oos_z = utils.restore_state(restore_filepath, restore_run_ID)
    print('RESTARTING FROM STORED STATE: ' + restore_run_ID)
    print('# electrons at restart:', Ne)
    print('E_now:', E_now)
    print('interation codes (before reset):', ints)
else:
    z_ion = []
    z_rot_ex = []
    z_vib_ex = []
    z_B_ex = []
    z_C_ex = []
    z_a_ex = []
    z_b_ex = []
    z_c_ex = []
    z_e_ex = []
    z_exit_z = []
    z_exit_E = []
    E_exit_E = [] # energy of e- which exit due to low energy
    E_exit_z = [] # energy of e- which exit due to altitude
    E_thermalization = []
    z_thermalization = []
    N_oos_z = 0
    N_oos_E = 0
    z_now = np.ones(Ne) * H0
    oos_idxs = np.zeros(Ne).astype(int)
    Ne_tot = 0 # track total number of incident + secondary electrons, including those that have exited the simulation
    E_now = np.ones(Ne) * e0
    E = np.ones(Ne) * e0
    v_z_now = - np.sign(cos_theta_0) * np.ones(Ne) * np.sqrt(2*e0/m)  # NB: positve cos(theta) means downwards.
    v_h_now = np.zeros(Ne) 
    cos_theta = np.ones(Ne) * cos_theta_0
    i = 0
z_next = np.ones(Ne) * np.nan
E_next = np.zeros(Ne) * np.nan
v_z_next = np.zeros(Ne) * np.nan
v_h_next = np.zeros(Ne) * np.nan
ints = np.zeros(Ne) * np.nan
    
if restore_state:
    run_ID = restore_run_ID + '_RESTORED'
else:
    run_ID = 'Ne=' + str(Ne) + 'E0=' + str(e0_kev) + 'cos(theta)=' + str(cos_theta_0) + '_' + atm_type + '_' + version +  '_' + t

if diagnostics:
    Ne_t = [] 
    Ne_t += [Ne]
    ti = tm.perf_counter()
    print('Run ID:', run_ID)
    print('Logfile:', logfile)
    print()
    print('Initial # electrons:', Ne)
    print('Storing minimum data:', args.minimum_store)


# initial timestep
nH2 = get_n_H2(z_now) 
xsec1 = ionization_xsec(E_now)
xsec2 = rot_excitation_xsec(E_now)
xsec3 = elastic_scat_xsec(E_now)
xsec4 = B_excitation_xsec(E_now)
xsec5 = C_excitation_xsec(E_now)
xsec6 = a_excitation_xsec(E_now)
xsec7 = b_excitation_xsec(E_now)
xsec8 = c_excitation_xsec(E_now)
xsec9 = e_excitation_xsec(E_now)
xsec10 = vib_excitation_xsec(E_now)
xsec_tot = xsec1 + xsec2 + xsec3 + xsec4 + xsec5 + xsec6 + xsec7 + xsec8 + xsec9 + xsec10
P_max = 0.08 # maximum P we are comfortable with in one step
dt = np.abs(np.log(1-P_max) / (nH2 * xsec_tot * v_z_now)) # to keep the first values of P < 0.1 (divide by 10 to be extra sure)
toa_scale_height = k*T/(mH2*g) # top of atm scale height for enforcing small steps
max_allowed_step = toa_scale_height/10
if (np.abs(v_z_now * dt) > max_allowed_step).any():
    dt = np.abs(max_allowed_step / v_z_now)
    
print("Maximum allowed interaction probability:", P_max)
if diagnostics:
    print('Initial cross sections:')
    print('   Total       :', xsec_tot[0])
    print('   Rotational  :', rot_excitation_xsec(E_now)[0])
    print('   B excitation:', B_excitation_xsec(E_now)[0])
    print('   C excitation:', C_excitation_xsec(E_now)[0])
    print('   a excitation:', a_excitation_xsec(E_now)[0])
    print('   b excitation:', b_excitation_xsec(E_now)[0])
    print('   c excitation:', c_excitation_xsec(E_now)[0])
    print('   e excitation:', e_excitation_xsec(E_now)[0])
    print('   Vibrational :', vib_excitation_xsec(E_now)[0])
    print('   Elastic     :', elastic_scat_xsec(E_now)[0])
    print('Initial H2 number density:', get_n_H2(H0))
    print('Starting altitude:', H0)
    print('Initial z velocity:', v_z_now[0])
    print('Initial dt:', dt[0])

# generate many random numbers before entering loop for speed (will generate more as needed)
rand_nums, N_rand, counter = utils.generate_rands(100*Ne)

# fraction that have left simulation
frac_exit = 0

# start loop runtime tracker -- save intermediates each hour and reset t0
intermediate_t0 = tm.time() # seconds
stopping_criterion = False
print()
print('Entering simulation loop.')
while stopping_criterion == False:
    
    # add more random numbers if needed
    if counter + 4*Ne + 4 >= N_rand: # maximum that could be needed on this loop: 4*Ne + 4
        print('step:', i)
        print('current Ne:', Ne)
        print(frac_exit, ' of running total electrons have exited main loop.')
        print('generating', len(rand_nums), 'new random values')
        rand_nums, N_rand, counter = utils.generate_rands(40*Ne + 40)

    # just for diagnostics
    dt_prev = np.copy(dt)
    
    # calculate prelimonary dz for this step
    # for electrons that have left the simulation due to reaching the max height, do not allow z to change
    toa_scale_height = k*T/(mH2*g) # top of atm scale height for enforcing small steps
    max_allowed_step = toa_scale_height/10
    dt[np.abs(v_z_now*dt) > max_allowed_step] = np.abs(max_allowed_step/v_z_now[np.abs(v_z_now*dt) > max_allowed_step]) # enforce steps smaller than
    z_step = v_z_now*dt
    zf_p = z_now + z_step
    dz = zf_p - z_now 

    # calculate prelimonary column density over this step
    alpha = np.log(get_n_H2(zf_p)/get_n_H2(z_now)) / (zf_p - z_now)
    A = get_n_H2(z_now)
    column_density = np.abs(-A * (np.exp(alpha*(zf_p - z_now)) - 1)/(alpha*cos_theta))
    idx_small = np.abs(cos_theta) <= 1e-3
    column_density[idx_small] = np.abs(get_n_H2(z_now[idx_small]) * v_h_now[idx_small] * dt[idx_small])
    v = np.sqrt(2*E_now/m)

    # pick whether a collision occured    
    P = 1 - np.exp(-xsec_tot*column_density)     
    R = rand_nums[counter:counter+Ne]
    counter += Ne 
    coll = R < P
    
    # check for errors
    if (P > 0.1).any():
        print('P exceeds 0.1, timestep too large, P =', P, ' at ', i, 'th step')
        print('     P = ', P[P > 0.1], ' for electron # ', np.where(P > 0.1))
        break
    if (np.isnan(dt)).any():
        print('Error: dt is Nan.')
        print('dt =', dt, ' at ', i, 'th step')
        print('     dt = ', dt[np.isnan(dt)], ' for electron # ', np.where(np.isnan(dt))
        break     

    # pick where along path the collision occured. Actaully, we don't care where it happened 
    # in the x-y plane, so we can collapse it and only consider the vertical motion.
    signed_cos_theta = -np.abs(cos_theta)*np.sign(dz) # enforce proper sign
    z_next[coll] = (1/alpha[coll])*np.log((alpha[coll]*signed_cos_theta[coll]*np.log(1-R[coll])) / (A[coll]*xsec_tot[coll]) + 1) + z_now[coll]
    z_next[idx_small] = z_now[idx_small] + dz[idx_small] * np.log(1-R[idx_small])/np.log(1-P_max) # Taylor expansion approximation for small cos(theta)
    z_next[~coll] = zf_p[~coll] # set final z for electrons which had no collision

    # for e- which had a collision, pick which type (P's all sum to 1)
    n_coll = len(z_now[coll]) # max number of collisions would be Ne
    P_ion = xsec1/xsec_tot # P(ionization)
    P_rot_exci = xsec2/xsec_tot # P(rotational excitation)
    P_vib_exci = xsec10/xsec_tot # P(rotational excitation)
    P_B_exci = xsec4/xsec_tot # P(B excitation)
    P_C_exci = xsec5/xsec_tot # P(C excitation)
    P_a_exci = xsec6/xsec_tot # P(a excitation)
    P_b_exci = xsec7/xsec_tot # P(b excitation)
    P_c_exci = xsec8/xsec_tot # P(c excitation)
    P_e_exci = xsec9/xsec_tot # P(d excitation)
    P_el = xsec3/xsec_tot # P(elastic scattering)
    type_picker = rand_nums[counter:counter+n_coll] 
    counter += n_coll
    types = np.ones(n_coll) # first assign all to ionization
    types[type_picker >= P_ion[coll]] = 2 # assign the rest to rotational excitation 
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll])] = 3 # assign the rest to elastic scattering
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll])] = 4 # assign the rest to B excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll]] = 5 # assign the rest to C excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll] + P_C_exci[coll]] = 6 # assign the rest to a excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll] + P_C_exci[coll] + P_a_exci[coll]] = 7 # assign the rest to b excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll] + P_C_exci[coll] + P_a_exci[coll] + P_b_exci[coll]] = 8 # assign the rest to c excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll] + P_C_exci[coll] + P_a_exci[coll] + P_b_exci[coll] + P_c_exci[coll]] = 9 # assign the rest to d excitation
    types[type_picker >= (P_ion[coll] + P_rot_exci[coll] + P_el[coll]) + P_B_exci[coll] + P_C_exci[coll] + P_a_exci[coll] + P_b_exci[coll] + P_c_exci[coll] + P_e_exci[coll]] = 10 # assign the rest to vibrational excitation

    # update interaction codes
    ints[coll] = types
    ints[~coll] = 0   

    # define when to use each type of scattering angle distribution
    idxs_rutherford = (ints == 1) + (ints == 3) + (ints == 4) + (ints == 5)
    idxs_isotropic = (ints == 2) + (ints == 6) + (ints == 7) + (ints == 8) + (ints == 9) + (ints == 10)
    ionized_idxs = ints == 1
    N_rutherford = sum(idxs_rutherford)
  
    # update energies
    E_next[ints == 0] = E_now[ints == 0] # no interaction
    E_ej = utils.vahedi_ejected_energy(E_now[ints == 1], E_ion) # secondaries
    E_next[ints == 1] = E_now[ints == 1] - E_ion - E_ej  # ionizations     
    E_next[ints == 2] = E_now[ints == 2] - E_rot_excitation # rot excitations
    E_next[ints == 10] = E_now[ints == 10] - E_vib_excitation # vib excitations
    E_next[ints == 4] = E_now[ints == 4] - E_B_excitation # B excitations
    E_next[ints == 5] = E_now[ints == 5] - E_C_excitation # C excitations
    E_next[ints == 6] = E_now[ints == 6] - E_a_excitation # a excitations
    E_next[ints == 7] = E_now[ints == 7] - E_b_excitation # b excitations
    E_next[ints == 8] = E_now[ints == 8] - E_c_excitation # c excitations
    E_next[ints == 9] = E_now[ints == 9] - E_e_excitation # e excitations
    E_next[ints == 3] = E_now[ints == 3] # elastic scatterings

    # don't let energy be negative    
    E_next[E_next < 0] = E_rot_excitation/10 # hacky, but will cause problems if set to zero (WHY SHOULD THIS EVER HAPPEN THOUGH)

    # calculate velocities 
    v_scat_rutherford = np.sqrt(2*E_next[idxs_rutherford]/m) # final velocity after scattering for electrons with rutherford scattering, t = i+1 
    v_scat_isotropic = np.sqrt(2*E_next[idxs_isotropic]/m) # final velocity after scattering for electrons with rutherford scattering, t = i+1 

    # calculate scattering angles for Rutherford scattered electrons relative to incident e-
    cdf_val = rand_nums[counter:counter+N_rutherford]
    counter += N_rutherford
    scat_theta = utils.pick_theta_v3(E_now[idxs_rutherford],m,cdf_val) # screened rutherford 
    scat_phi  = rand_nums[counter:counter+N_rutherford] * 2*np.pi # angle around circle of possible final trajectories 
    counter += N_rutherford        

    # update velocities for Rutherford scattered electrons
    theta_i = np.arccos(cos_theta[idxs_rutherford])
    cos_theta[idxs_rutherford] = np.sin(scat_theta)*np.sin(scat_phi)*np.sin(theta_i) + np.cos(scat_theta)*np.cos(theta_i)
    v_z_next[idxs_rutherford] = -v_scat_rutherford * cos_theta[idxs_rutherford]
    v_h_next[idxs_rutherford] = np.sqrt(v_scat_rutherford**2 - v_z_next[idxs_rutherford]**2) # can only be positive 

    # don't change velocities for electrons with no interaction (cos_theta array is unchanged for these indices)
    v_z_next[ints == 0] = v_z_now[ints == 0]
    v_h_next[ints == 0] = v_h_now[ints == 0]

    # also dont change velocities for electrons that exited due to low energy (cos_theta array is unchanged for these indices)
    v_z_next[ints == -2] = v_z_now[ints == -2]       
    v_h_next[ints == -2] = v_h_now[ints == -2]   

    # update velocities for isotropically scattered electrons
    v_theta = rand_nums[counter:counter+1] * np.pi # for now assume isotropic NOTE: should be ok to pick the same number for each
    v_phi = rand_nums[counter:counter+1] * 2*np.pi # for now assume isotropic
    counter += 2
    theta_i = np.arccos(cos_theta[idxs_isotropic])
    cos_theta[idxs_isotropic] = np.sin(v_theta)*np.sin(v_phi)*np.sin(theta_i) + np.cos(v_theta)*np.cos(theta_i)
    v_z_next[idxs_isotropic] = -v_scat_isotropic * cos_theta[idxs_isotropic]
    v_h_next[idxs_isotropic] = np.sqrt(v_scat_isotropic**2 - v_z_next[idxs_isotropic]**2) # can only be positive 

    # create ionization electrons
    E_creation = E_ej # E1_secondary * E_now[ionized_idxs]
    v_creation = np.sqrt(2*E_creation/m)
    v_theta_creation = rand_nums[counter:counter+1] * np.pi # for now assume isotropic NOTE: should be ok to pick the same number for each
    v_phi_creation = rand_nums[counter:counter+1] * 2*np.pi # for now assume isotropic
    counter += 2
    v_x_creation = v_creation*np.sin(v_theta_creation)*np.cos(v_phi_creation)
    v_y_creation = v_creation*np.sin(v_theta_creation)*np.sin(v_phi_creation)
    v_z_creation = -v_creation*np.cos(v_theta_creation)
    v_h_creation = np.sqrt(v_x_creation**2 + v_y_creation**2)
    cos_theta_creation = -v_z_creation / np.sqrt(v_z_creation**2 + v_h_creation**2)

    # append new electrons
    Ni = sum(ionized_idxs)
    z_add = z_next[ionized_idxs]
    ints_add = np.ones((Ni))*(-1)
    ints = np.hstack([ints, ints_add])
    z_next = np.hstack([z_next, z_add])                              
    E_next = np.hstack([E_next, E_creation])
    v_z_next = np.hstack([v_z_next, v_z_creation])
    v_h_next = np.hstack([v_h_next, v_h_creation])
    cos_theta = np.hstack([cos_theta, cos_theta_creation])
    Ne += Ni   
    Ne_tot += Ni

    # check which have left simulation
    oos_z_idxs = z_next > max_height
    oos_E_idxs = E_next < E_b_excitation # E_threshold
    oos_idxs = oos_z_idxs + oos_E_idxs 
    ints[oos_z_idxs] = -3
    ints[oos_E_idxs] = -2
    N_oos_z += sum(oos_z_idxs)
    N_oos_E += sum(oos_E_idxs)

    # record the heights at which each interaction type occured
    z_ion += list(z_next[ints == 1])
    z_rot_ex += list(z_next[ints == 2])
    z_vib_ex += list(z_next[ints == 10])
    z_B_ex += list(z_next[ints == 4])
    z_C_ex += list(z_next[ints == 5])
    z_a_ex += list(z_next[ints == 6])
    z_b_ex += list(z_next[ints == 7])
    z_c_ex += list(z_next[ints == 8])
    z_e_ex += list(z_next[ints == 9])   
    z_exit_z += list(z_next[ints == -3])
    z_exit_E += list(z_next[ints == -2])

    # record the energies of the electrons exiting due to low energy or altitude
    E_exit_E += list(E_next[ints == -2])
    E_exit_z += list(E_next[ints == -3])

    # Drop the electrons which have exited out of the simulation
    keep_idxs = np.invert(oos_idxs)
    z_next = z_next[keep_idxs]
    E_next = E_next[keep_idxs]
    v_z_next = v_z_next[keep_idxs]
    v_h_next = v_h_next[keep_idxs]
    cos_theta = cos_theta[keep_idxs]
    Ne -= sum(oos_idxs)

    # determine cross sections going into the next timestep
    nH2 = get_n_H2(z_next) 
    xsec1 = ionization_xsec(E_next)
    xsec2 = rot_excitation_xsec(E_next)
    xsec3 = elastic_scat_xsec(E_next)
    xsec4 = B_excitation_xsec(E_next)
    xsec5 = C_excitation_xsec(E_next)
    xsec6 = a_excitation_xsec(E_next)
    xsec7 = b_excitation_xsec(E_next)
    xsec8 = c_excitation_xsec(E_next)
    xsec9 = e_excitation_xsec(E_next)
    xsec10 = vib_excitation_xsec(E_next)
    xsec_tot = xsec1 + xsec2 + xsec3 + xsec4 + xsec5 + xsec6 +xsec7 + xsec8 + xsec9 + xsec10

    # set dt for the next timestep 
    v = np.sqrt(2*E_next/m)  
    dt = np.abs(np.log(1-P_max)/(-nH2*xsec_tot*v))  # dt < ln(0.9)/ (-utils.n_H2(z[:,i])*xsec_tot*v) 

    frac_exit_prev = frac_exit
    frac_exit = N_oos_E/(len(E_next) + N_oos_E)
    stopping_criterion = len(E_next) == 0 # frac_exit > exit_pcnt #> 0.99
    if stopping_criterion:
        print('')
        print('Main simulation has ended successfully (all e- have energy < ' + str(E_threshold) + ').')
        print('')
    else:          
        # Update for next iteration
        E_now = np.copy(E_next)
        z_now = np.copy(z_next)
        v_z_now = np.copy(v_z_next)
        v_h_now = np.copy(v_h_next)
        E_next = np.zeros(Ne) * np.nan
        v_z_next = np.zeros(Ne) * np.nan
        v_h_next = np.zeros(Ne) * np.nan
        ints = np.zeros(Ne) * np.nan
   
        
    # Every 2 hours of loop runtime, save intermediate results for restarting
    intermediate_tf = tm.time() # seconds
    if (intermediate_tf - intermediate_t0)/60**2 > 2:
        tf = tm.perf_counter()
        print()
        print('Reached another interval runtime, saving intermediate state.')
        print('Current runtime:', tf - ti, 's')
        print('Current fraction thermalized:', frac_exit)
        print('Current step:', i)
        print('Median vertical distance traveled:', np.nanmedian(H0 - z_now/1000), 'km ')
        print('Current total electrons used in simulation:', Ne_tot)
        print('Current total electrons remaining in simulation:', Ne)
        print('Median dt values:', np.nanmedian(dt), 's')
        print('Overwriting any previously saved intermediate outputs for this run.')

        # save outputs
        if vary_theta:
            output_dir = './' + str(e0_kev) + 'keV_costheta=' +  str(cos_theta_0)
        else:
            output_dir = './' + str(e0_kev) + 'keV'
        if os.path.exists(output_dir):
            print()
            this_dir = os.getcwd()
            print('WARNING: saving to directory' + this_dir + '/' + output_dir + ' which already exists. No files (for OTHER runs) will be overwritten because each output is saved with a unique ID, but perhaps you meant to submit this job in a new directory?')
            print()
        else:
            os.mkdir('./' + output_dir)
        #
        utils.save_state_min_store(E_now, cos_theta, dt, ints, z_now, v_z_now, v_h_now, Ne, Ne_tot, i, N_oos_E, N_oos_z, output_dir, 'state_' + run_ID + '.pickle')
        utils.save_results_min_store(z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, z_thermalization, E_thermalization, output_dir, 'results_' + run_ID + '.h5')        
        intermediate_t0 = tm.time()
        
    i += 1

# Now, the rest of the interactions are either rotational or vibrational excitations, and the electrons are not going to move significantly
# from their current locations.
# First, iterate the electrons from their current energies (< ~4.5eV) until below the threshold for vibrational excitation, ~0.9eV
E_remaining = np.array([])
z_remaining = np.array([])
j = 0
E = np.array(E_exit_E) # the energies and altitudes of the electrons when then exited the main loop due to being below the ~4.5eV threshold
z = np.array(z_exit_E)
print()
print('Entering second simulation loop for low energy electrons.')
while len(E) > 0:
    Ne = len(E)
    # calculate probabilities
    xsec_rot = rot_excitation_xsec(E)
    xsec_vib = vib_excitation_xsec(E)
    xsec = xsec_rot + xsec_vib
    P_rot  = xsec_rot / xsec
    P_vib = xsec_vib / xsec
    R = np.random.uniform(0,1,Ne)
    # record where interactions happened (z is not changing, so record electron location)
    z_rot_ex += list(z[R < P_rot])
    z_vib_ex += list(z[R >= P_rot])
    # udpate energies
    E[R < P_rot] = E[R < P_rot] - E_rot_excitation
    E[R >= P_rot] = E[R >= P_rot] - E_vib_excitation
    # remove low energy electrons, and record their energies and altitudes
    idxs = E>E_vib_excitation
    E_remaining = np.append(E_remaining, E[np.invert(idxs)])
    z_remaining = np.append(z_remaining, z[np.invert(idxs)])
    E = E[idxs]
    z = z[idxs]
    j+=1 
              
# Count up how many rotational excitations can happen with the remaining energy, and how much energy will be
#        left over below the rotational excitation threshold
# All interactions now will be rotational excitations only because we are below the vibrational excitation threshold
print('Entering final step to determine lowest energy interactions.')
N_rot_remaining = np.floor(E_remaining / E_rot_excitation).astype(int)
E_thermalization = E_remaining % E_rot_excitation
z_thermalization = z_remaining # we didn't let them move
z_rot_remaining = np.repeat(np.array(z_remaining), N_rot_remaining) # repeat the altitudes of the remaining electrons by the number of rotational excitations each had.
z_rot_ex += list(z_rot_remaining) 


print() 
if diagnostics:
    tf = tm.perf_counter()
    print('Runtime:', tf - ti, 's')
    print('Final fraction thermalized:', frac_exit)
    print('Final number of steps:', i)
    print('Median vertical distance traveled:', np.nanmedian(H0 - z_now/1000), 'km ')
    print('Final total electrons used in simulation:', Ne_tot)
    print('Final total electrons remaining in simulation:', Ne)

    # save outputs
    if vary_theta:
        output_dir = './' + str(e0_kev) + 'keV_costheta=' +  str(cos_theta_0)
    else:
        output_dir = './' + str(e0_kev) + 'keV'
    if os.path.exists(output_dir):
        print()
        this_dir = os.getcwd()
        print('WARNING: saving to directory' + this_dir + '/' + output_dir + ' which already exists. No files will be overwritten because each output is saved with a unique ID, but perhaps you meant to submit this job in a new directory?')
        print()
    else:
        os.mkdir('./' + output_dir)
    #
    utils.save_state_min_store(E_now, cos_theta, dt, ints, z_now, v_z_now, v_h_now, Ne, Ne_tot, i, N_oos_E, N_oos_z, output_dir, 'state_' + run_ID + '.pickle')
    utils.save_results_min_store(z_ion, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, E_exit_z, z_thermalization, E_thermalization, output_dir, 'results_' + run_ID + '.h5')

    # move logfile
    os.rename(logfile, output_dir + '/logfile.txt')
        
  
                       