# NB: corresponds to version 21 in working copy code

# Interaction codes
# '-3' --> particle exit simulation do to reaching max height
# '-2' --> particle exit simulation do to low energy
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
parser.add_argument('--Nsteps',
                    dest='N_steps',
                    type=int,
                    help='Max number of steps simulate, or, if starting from restored state, number of additioanl steps simulate.',
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
parser.add_argument('--sonora_filepath',
                    dest='sonora_filepath',
                    type=str,
                    help='Relative path to Sonora atmospheric profiles.',
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
parser.add_argument('--logfile',
                    dest='logfile',
                    type=str,
                    help='Name of run logfile.')

args = parser.parse_args()
Ne = args.Ne0
e0 = args.e0
e0 = e0 * u.keV
N_steps = args.N_steps
cos_theta_0 = args.cos_theta
atm_type = args.atm_type
diagnostics = args.diagnostics
save_state = args.save_state
vary_theta = args.vary_theta
logfile = args.logfile
sonora_filepath = args.sonora_filepath

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
E_threshold = 1.60218e-19 # 1 eV in J, energy for which ionization cross section equation doesn't hold
E_ion = 15.43 * u.eV # eV to ionize H2 (NIST "binding energy")

# pick which density profile to use
if atm_type == 'Jupiter': 
    g = 24.79 * u.m/u.s**2 # NASA Jupiter fact sheet...
    H0 = 2000e3 # [m]  2000 "top" of atmosphere altitude [km], P ~ 10^-13 bars
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 150.78 # [K] T at top of atm from Sonora, 10^-8 bars
    from utils import rot_excitation_xsec_Jupiter as rot_excitation_xsec
    get_n_H2, get_P_H2 = utils.construct_profile_Jupiter(atm_type, max_height, sonora_filepath) # uses Galileo data
elif atm_type == 'T1400_g4.0':
    g = 100 * u.m/u.s**2  # log g = 4.0 in cgs --> g = 1e4 in cgs cm/s^2 --> g = 1e2 in SI m/s^2
    H0 = 755.560e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 550.72 # [K] T at top of atm from Sonora, 10^-8 bars
    from utils import rot_excitation_xsec_1400K as rot_excitation_xsec
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T1400_g5.0':
    from utils import rot_excitation_xsec_1400K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2  # log g = 4.0 in cgs --> g = 1e4 in cgs cm/s^2 --> g = 1e2 in SI m/s^2
    H0 =  71.970e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 575.509596 # [K] T at top of atm from Sonora, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T900_g5.0':
    H0 = 37.612e3 # [m]  alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs --> g = 1e5 in cgs cm/s^2 --> g = 1e3 in SI m/s^2
    T = 283.47 # [K] T at top of atm from Sonora, 10^-8 bars
    from utils import rot_excitation_xsec_900K as rot_excitation_xsec
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T900_g4.0':
    from utils import rot_excitation_xsec_900K as rot_excitation_xsec
    g = 100 * u.m/u.s**2 # log g = 5.0 in cgs --> g = 1e5 in cgs cm/s^2 --> g = 1e3 in SI m/s^2
    H0 = 398.090e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 263.422216 # [K] T at top of atm from Sonora, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T482_g4.7':
    from utils import rot_excitation_xsec_482K as rot_excitation_xsec
    g = 501.187 * u.m/u.s**2 # log g = 5.0 in cgs --> g = 1e5 in cgs cm/s^2 --> g = 1e3 in SI m/s^2
    H0 = 37.942e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 144.485 # [K] T at top of atm from Sonora, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T2000_g5.0':
    from utils import rot_excitation_xsec_2000K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs --> g = 1e5 in cgs cm/s^2 --> g = 1e3 in SI m/s^2
    H0 = 92.236e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 717.83 # [K] T at top of atm from Sonora, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)
elif atm_type == 'T500_g5.0':
    from utils import rot_excitation_xsec_500K as rot_excitation_xsec
    g = 1000 * u.m/u.s**2 # log g = 5.0 in cgs --> g = 1e5 in cgs cm/s^2 --> g = 1e3 in SI m/s^2
    H0 = 19.4816e3 # [m] alttiude corresponding to isothermally extended partial pressure of H2 of approx. 1e-13 bar
    max_height = 1.2 * H0 # consider electron to be out of the simulation above this
    T = 145.31 # [K] T at top of atm from Sonora, 10^-8 bars
    get_n_H2, get_P_H2 = utils.construct_profiles(atm_type, max_height, sonora_filepath)    
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

# initialize saved quantities
z_ion = []
z_el_scat = []
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
N_oos_z = 0
N_oos_E = 0
z_now = np.ones(Ne) * H0
z_next = np.ones(Ne) * np.nan
ints = np.ones(Ne) * np.nan
oos_idxs = np.zeros(Ne).astype(int)
Ne_tot = 0 # track total number of incident + secondary electrons, including those that have exited the simulation

    
# for saving
version = 'v21'
run_ID = 'Nsteps=' + str(N_steps) + 'Ne=' + str(Ne) + 'E0=' + str(e0_kev) + 'cos(theta)=' + str(cos_theta_0) + '_' + atm_type + '_' + version +  '_' + t
if diagnostics:
    Ne_t = np.zeros(N_steps)
    Ne_t[i0] = Ne
    ti = tm.perf_counter()
    if not restore_state:
        print('Run ID:', run_ID)
        print()
        print('Initial # electrons:', Ne)
        print('Max allowed total steps:', N_steps)
        print('Storing minimum data:', args.minimum_store)

nH2 = get_n_H2(z_now) 
    
# initial timestep (taking 0th electron since they are all the same at t=0)
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
xsec_tot_0 = xsec_tot[0] # P = 1 - exp(dt * stuff) --> dt = ln(1-P)/stuff --> P1 = N*P0 --> dt1 = ln(1-N*P0)/stuff 
P_max = 0.08 # maximum P we are comfortable with in one step
dt = np.abs(np.log(1-P_max) / (nH2 * xsec_tot * v_z_now)) # to keep the first values of P < 0.1 (divide by 10 to be extra sure)
print("Maximum allowed interaction probability:", P_max)
if diagnostics:
    print('Initial cross sections:')
    print('   Total       :', xsec_tot_0)
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
    print('Initial z velocity:', v_z_now[0])
    print('Initial dt:', dt[0])

# initialize tracked quantities
E_next = np.zeros(Ne) * np.nan
v_z_next = np.zeros(Ne) * np.nan
v_h_next = np.zeros(Ne) * np.nan
frac_exit = 0

# generate many random numbers before entering loop for speed (will generate more as needed)
rand_nums, N_rand, counter = utils.generate_rands(100*Ne)

print()
print('Entering simulation loop.')
try:
    for i in range(i0, N_steps-1): 

        # add more random numbers if needed
        if counter + 4*Ne + 4 >= N_rand: # maximum that could be needed on this loop: 4*Ne + 4
            print('step:', i)
            print('generating', len(rand_nums), 'new random numbers')
            print('current Ne:', Ne)
            rand_nums, N_rand, counter = utils.generate_rands(40*Ne + 40)

        # just for diagnostics
        dt_prev = dt
        v_z_now_prev = v_z_now

        # calculate prelimonary dz for this step
        # for electrons that have left the simulation due to reaching the max height, do not allow z to change
        toa_scale_height = k*T/(mH2*g) # top of atm scale height for enforcing small steps
        max_allowed_step = toa_scale_height/10
        dt[np.abs(v_z_now*dt) > max_allowed_step] = np.abs(max_allowed_step/v_z_now[np.abs(v_z_now*dt) > max_allowed_step]) # enforce steps smaller than
        z_step = v_z_now*dt
        zf_p = z_now + z_step
        dz = zf_p - z_now 

        # calculate preliminary column density over this step
        alpha = np.log(get_n_H2(zf_p)/get_n_H2(z_now)) / (zf_p - z_now)
        A = get_n_H2(z_now)
        column_density = np.abs(-A * (np.exp(alpha*(zf_p - z_now)) - 1)/(alpha*cos_theta))
        column_density[cos_theta == 0] = np.abs(get_n_H2(z_now[cos_theta == 0]) * v_h_now[cos_theta == 0] * dt[cos_theta == 0])
 
        # pick whether a collision occured    
        v = np.sqrt(2*E_now/m)
        P = 1 - np.exp(-xsec_tot*column_density) 
        if (P > 0.1).any():
            print('WARNING: P exceeds 0.1, timestep too large, P =', P, ' at ', i, 'th step')

        R = rand_nums[counter:counter+Ne]
        counter += Ne
        coll = R < P

        # pick where along path the collision occured. Actaully, we don't care where it happened 
        # in the x-y plane, so we can collpse it and only consider the vertical motion.
        signed_cos_theta = -np.abs(cos_theta)*np.sign(dz) # enforce proper sign
        z_next[coll] = (1/alpha[coll])*np.log((alpha[coll]*signed_cos_theta[coll]*np.log(1-R[coll])) / (A[coll]*xsec_tot[coll]) + 1) + z_now[coll]
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
        E_next[E_next < 0] = E_threshold/10 # hacky, but will cause problems if set to zero

        # calculate velocities 
        v_scat_rutherford = np.sqrt(2*E_next[idxs_rutherford]/m) # final velocity after scattering for electrons with rutherford scattering, t = i+1 
        v_scat_isotropic = np.sqrt(2*E_next[idxs_isotropic]/m) # final velocity after scattering for electrons with rutherford scattering, t = i+1 

        # calculate scattering angles for Rutherford scattered electrons relative to incident e-
        cdf_val = rand_nums[counter:counter+N_rutherford]
        counter += N_rutherford
        scat_theta = utils.pick_theta(E_now[idxs_rutherford],m,cdf_val) # screened rutherford 
        scat_phi  = rand_nums[counter:counter+N_rutherford] * 2*np.pi # angle around circle of possible final trajectories 
        counter += N_rutherford

        # update velocities for Rutherford scattered electrons
        # 1) find angle of current (prior to interaction) v vector in original axes
        v = np.sqrt(v_z_now[idxs_rutherford]**2 + v_h_now[idxs_rutherford]**2)
        v_theta = np.pi - np.arccos(v_z_now[idxs_rutherford]/v) # formula from wikipedia uses the supplementary angle
        # 2) rotate the scattering direction vector by the reverse of those angles
        v_scat_x = v_scat_rutherford*np.sin(scat_theta)*np.cos(scat_phi) # x component in frame of incident e-
        v_scat_y = v_scat_rutherford*np.sin(scat_theta)*np.sin(scat_phi) # y component in frame of incident e-
        v_scat_z = -v_scat_rutherford*np.cos(scat_theta) # z component in frame of incident e-
        v_scat_z_lab, v_scat_h_lab = utils.transform_collapsed(v_scat_x, v_scat_y, v_scat_z, dtheta = v_theta) # by 2*np.pi - phi and np.pi - theta
        v_z_next[idxs_rutherford] = v_scat_z_lab
        v_h_next[idxs_rutherford] = v_scat_h_lab
        cos_theta[idxs_rutherford] = -v_z_next[idxs_rutherford] / np.sqrt(v_z_next[idxs_rutherford]**2 + v_h_next[idxs_rutherford]**2)

        # don't change velocities for electrons with no interaction (cos_theta array is unchanged for these indices)
        v_z_next[ints == 0] = v_z_now[ints == 0]
        v_h_next[ints == 0] = v_h_now[ints == 0]

        # also dont change velocities for electrons that exited due to low energy (cos_theta array is unchanged for these indices)
        v_z_next[ints == -2] = v_z_now[ints == -2]       
        v_h_next[ints == -2] = v_h_now[ints == -2]   

        # update velocities for isotropically scattered electrons
        v_theta = rand_nums[counter:counter+1] * np.pi # for now assume isotropic NOTE: I think it's ok to pick the same number for each
        v_phi = rand_nums[counter:counter+1] * 2*np.pi # for now assume isotropic
        counter += 2
        v_x = v_scat_isotropic*np.sin(v_theta)*np.cos(v_phi)
        v_y = v_scat_isotropic*np.sin(v_theta)*np.sin(v_phi)
        v_z_next[idxs_isotropic] = -v_scat_isotropic*np.cos(v_theta)
        v_h_next[idxs_isotropic] = np.sqrt(v_x**2 + v_y**2)
        cos_theta[idxs_isotropic] = -v_z_next[idxs_isotropic] / np.sqrt(v_z_next[idxs_isotropic]**2 + v_h_next[idxs_isotropic]**2)

        # create ionization electrons
        E_creation = E_ej # E1_secondary * E_now[ionized_idxs]
        v_creation = np.sqrt(2*E_creation/m)
        v_theta_creation = rand_nums[counter:counter+1] * np.pi # for now assume isotropic NOTE: I think it's ok to pick the same number for each
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
        oos_E_idxs = E_next < E_threshold
        oos_idxs = oos_z_idxs + oos_E_idxs
        ints[oos_z_idxs] = -3
        ints[oos_E_idxs] = -2
        N_oos_z += sum(oos_z_idxs)
        N_oos_E += sum(oos_E_idxs)

        # record the heights at which each interaction type occured
        z_ion += list(z_next[ints == 1])
        z_el_scat += list(z_next[ints == 3])
        z_rot_ex += list(z_next[ints == 2])
        z_vib_ex += list(z_next[ints == 10])
        z_B_ex += list(z_next[ints == 4])
        z_C_ex += list(z_next[ints == 5])
        z_a_ex += list(z_next[ints == 6])
        z_b_ex += list(z_next[ints == 7])
        z_c_ex += list(z_next[ints == 8])
        #z_d_ex += list(z_next[ints == 9])   
        z_e_ex += list(z_next[ints == 9])   
        z_exit_z += list(z_next[ints == -3])
        z_exit_E += list(z_next[ints == -2])

        # record the energies of the electrons exiting due to low energy
        E_exit_E += list(E_next[ints == -2])

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

        if (np.isnan(dt)).any():
            print()
            print('WARNING: Calculated Nan timestep value(s).')
            print('step number:', i)
   
        if np.isnan(z_next).any():
            print()
            print('WARNING: Calculated Nan altitude value(s).')
            print('step number:', i)

        # Update for next iteration
        E_now = E_next
        z_now = z_next
        v_z_now = v_z_next
        v_h_now = v_h_next
        E_next = np.zeros(Ne) * np.nan
        v_z_next = np.zeros(Ne) * np.nan
        v_h_next = np.zeros(Ne) * np.nan
        ints = np.zeros(Ne) * np.nan


        frac_exit_prev = frac_exit
        frac_exit = N_oos_E/(len(E_next + N_oos_E))
        if frac_exit > 0.98:
            print('')
            print('Simulation has ended successfully (>98% e- have energy < 1 eV).')
            print('')
            break

        if diagnostics:
            Ne_t[i] = Ne   

            
except Exception as err:
    print()
    print('An error has occured!')
    print('  Traceback:')
    traceback.print_tb(err.__traceback__)

print() 
    
if diagnostics:
    tf = tm.perf_counter()
    print('Runtime:', tf - ti, 's')
    print('Final fraction thermalized:', frac_exit)
    print('Final number of steps:', i)
    print('Median vertical distance traveled:', np.nanmedian(H0 - z_now/1000), 'km ')
    print('Final total electrons used in simulation:', Ne_tot)
    print('Final total electrons remaining in simulation:', Ne)
    print('Median dt values:', np.nanmedian(dt), 's')

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
    utils.save_state_min_store(E_now, cos_theta, dt, z_now,  output_dir, 'state_' + run_ID + '.h5')
    utils.save_results_min_store(z_ion, z_el_scat, z_rot_ex, z_vib_ex, z_B_ex, z_C_ex, z_a_ex, z_b_ex, z_c_ex, z_e_ex, z_exit_z, z_exit_E, E_exit_E, output_dir, 'results_' + run_ID + '.h5')

    # move logfile
    os.rename(logfile, output_dir + '/logfile.txt')
        
        
        
                       