import itertools
import numpy as np
import EDGE as edge
from astropy.io import ascii
import pdb
import pandas

'''
jobmaker.py

PURPOSE:
    Script that produces a file with all the parameters in a grid to run DIAD.
    It will also create a runall.csh script to run everything in the cluster.
    These files should be used with the python wrapper.

HOW TO USE THIS SCRIPT:
    1) Change the gridpath to be where you will create the parameters file.

    2) Change the paramfiletag to an insightful name that will help you identify
    the grid of models you ran three months from now.

    3) Change the parameters below to whatever set of parameters
    you want to run. The parameters should be surrounded by []'s and be
    separated by a commas (i.e., a list) if more than one value will be
    explored. If left as None, those parameters
    will not be set, and the default value in DIAD will be used. Mstar, Tstar,
    Rstar, DISTANCE, MDOT, MUI, ALPHA, RDISK, and AMAXB are necessary paramters
    and cannot be set to None.

    4) Change the labelname to the name of the object, e.g. 'gmauriga'

    5) Run the script

OPTIONAL PARAMETERS:
    fill: This will change the zero padding for jobnum in labelend.
    wall_model: [Boolean] Turns on/off all the disk components and just runs the
    wall. Generally set to 'False'
    Note: This is not looped over (i.e., all the models with this set to 'True'
    will only be walls).

NOTES:
    'AMAXW', 'AMAX' and/or 'AMAXB' only accept certain values. Here are the possible
    values:
       ['0.05', '0.1', '0.25', '0.5', '0.75', '1.0', '1.25',
        '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '10.0',
        '100','200','300','400',500', '600', '700', '800',' 900',
        '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000',
        '10000', '11000', '12000', '13000', '14000', '15000', '16000', '17000',
        '18000', '19000', '20000', '21000', '22000', '23000', '24000', '25000']

    There are two different ways of defining the max. grain size of the
    different dust populations to be used:
    1)  Define an AMAXB. The code will use 5 populations by default, with amax
        0.1, 1.0, 10.0, 100.0, and AMAXB. In this case, AMAX must be set to None
        and the grid will iterate through AMAXB if a list is given.
    2)  Define AMAX as a tuple. The code will use that set of max. grain sizes.
        In this case, AMAXB must be set to None.
'''

# Where you want the parameter file to be placed
gridpath = './'

#Tag that you can add to make the parameter file identifiable for a given run
#Can leave it blank if you don't care.
paramfiletag = 'gridX'

#Where you will be running the jobs on the cluster
clusterpath = '/projectnb/bu-disks/youruser/models/object/'+paramfiletag+'/'

#No need to add an underscore/jobnumber, the script will do that for you.
labelend = 'test'

#What number to start counting from, must be an integer
jobnumstart = 1

#Define parameters to feed into file
# If parameter is defined as list, it will iterate over it
params = {}
###########
# Necessary parameters (must give at least one value)
###########
params['MSTAR'] = 1.1 #Mass of the star in Msun
params['TSTAR'] = 4350 #Temperature of the star in K
params['RSTAR'] = 1.9 #Radius of the star in solar radii
params['DISTANCE'] = 160 #Distance to the star in pc
params['MDOT'] = 1e-8 #Mass accretion rate in the disk in solar masses per year
params['MUI'] = 0.5 #Cosine of the inclination of the disk.
params['ALPHA'] = 1e-3 # Viscosity in the disk
params['RDISK'] = 150 #Outer radius of the disk

# Define either AMAXB
params['AMAXB'] = 1000 #Maximum grain size in the midplane NOTE: Only accepts certain values
# Or define a full set of AMAX as a tuple (or list of tuples if you want to try
# different sets of AMAX). Example: params['AMAX'] = [(0.1,100), (0.1,1000)]
params['AMAX'] = None

###########
# Optional parameters (If left as None, it will use default value)
###########
params['D2G'] = 0.01 #Dust to gas mass ratio. (DEFAULT: 0.0065)
params['MDOTSTAR'] = None #Mass accretion rate at the star. (DEFAULT: same as MDOT)
params['AMAXW'] = None #Maximum grain size at the wall. (DEFAULT: 0.25)
params['ALPHASET'] = None # Turblent viscosity for settling (DEFAULT: same as ALPHA)
params['RC'] = None #Critial radius of tapered edge. If 2000, it will not have any effect. (DEFAULT: 2000)
params['GAMMA'] = None #Exponent of tapered edge. (DEFAULT: 2.0)
params['TEMP'] = None #Sublimation temperature of the grains (DEFAULT: 1400)
params['ALTINH'] = None #Height of the disk in scale heights. Can be scaled later. (DEFAULT: 1)
params['TSHOCK'] = None #Temperature of the shock. Usually left at 8000K (DEFAULT: 8000)
# For silicate composition (should add to 1)
params['AMORPFRAC_OLIVINE'] = 0.5 #Fraction of olivines (DEFAULT: 1)
params['AMORPFRAC_PYROXENE'] = 0.5 #Fraction of pyroxine (DEFAULT: 0)
params['FORSTERITE_FRAC'] = 0.0 #Fraction of fosterite (DEFAULT: 0)
params['ENSTATITE_FRAC']   = 0.0 #Fraction of enstatite (DEFAULT: 0)
#Gap creator:
#NOTE: these parameters themselves should be python tuples
#All used deltas need to have the same length as inter_r, so be careful with the combinations.
# DEFAULT: ()   (empty tuple)
params['INTER_R'] = None
params['RHO_DELTAS'] = None
params['TEMP_DELTAS'] = None
params['EPSBIG_DELTAS'] = None
params['EPS_DELTAS'] = None

# Non-iterative parameters
# These parameters will affect all the models
fill = 3 #Zero padding for the job numbers. Unless you are running many models (>1000) 3 is standard.
# Switches
wall_model = False #Set this to True if you want ONLY walls. (i.e., turns off the disk code).


#***********************************************
#Unlikely you need to change anything below here.
#***********************************************

# Checking that AMAX are well defined
if (params['AMAX'] != None) and (params['AMAXB'] != None):
    raise IOError('Define only a set of AMAX (with AMAXB=None) or '+
    ' AMAXB (with AMAX=None).')
if type(params['AMAX']) is list:
    if type(params['AMAX'][0]) is not tuple:
        raise IOError('AMAX should be defined as a tuple (if only one set of '+
        'AMAX is to be used), or a list of tuples (if more than one set of '+
        'AMAX will be tested.')
elif (params['AMAX'] != None) and (type(params['AMAX']) is not tuple):
    raise IOError('AMAX should be defined as a tuple (if only one set of '+
    'AMAX is to be used), or a list of tuples (if more than one set of '+
    'AMAX will be tested.')

# Which parameters we have to iterate through
iter_cols = []
iter_vals = []
relevant_cols = []
for param in params.keys():
    if type(params[param]) is list:
        if len(params[param]) > 1:
            iter_cols.append(param)
            iter_vals.append(params[param])
    elif params[param] != None:
        relevant_cols.append(param)

# Special cases
if params['INTER_R'] != None:
    relevant_cols.append('IMOD')
    params['IMOD'] = True
if wall_model:
    switches = ['IPHOT','IOPA','IVIS','IIRR','IPROP','ISEDT','ICOLLATE']
    relevant_cols.extend(switches)
    for switch in switches:
        params[switch] = False

grid_params = pandas.DataFrame(columns=['JobNum']+iter_cols+relevant_cols+['labelend'])
for ind,values in enumerate(itertools.product(*iter_vals)):
    row = {'JobNum':ind+jobnumstart}
    for i,param in enumerate(iter_cols):
        row[param] = values[i]
    for param in relevant_cols:
        row[param] = params[param]
    row['labelend'] = labelend+'_'+str(ind+jobnumstart).zfill(fill)
    grid_params = grid_params.append(row, ignore_index=True)

grid_params.to_csv(gridpath+paramfiletag+'_params_inp.csv', index=False)

if ind >= 1000:
    print('WARNING! GRID IS LARGE AND WILL USE SIGNIFICANT LIMITED COMPUTING '+
    'RESOURCES! SPEAK TO A SUPERVISOR BEFORE RUNNING ON THE CLUSTER!!!')
    warning = input('Please enter "I understand" to continue making this grid: ')
    if warning != 'I understand':
        raise ValueError('Grid creation canceled.')
    else:
        print('Continuing grid creation process...')

#Make a run all file for the grid you just created
edge.create_runall_py(jobnumstart, jobnumstart+ind,
paramfiletag+'_params_inp.csv', clusterpath, outpath = gridpath,
fill = fill)
