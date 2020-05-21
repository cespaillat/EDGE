import itertools
import numpy as np
import EDGE as edge
from astropy.io import ascii
import pdb

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

    3) Change the parameters in the brackets below to whatever set of parameters
    you want to run. The parameters should be surrounded by []'s and be
    separated by a commas (i.e., a list)

    4) Change the labelname to the name of the object, e.g. 'gmauriga'

    5) Run the script

OPTIONAL PARAMETERS:
    'amaxw' and 'mdotstar': Generally these take the value of 'amaxs' and 'mdot'
    respectively, but it is possible that you may want to change them.
    To do this, change them like you would any other parameter. If you DO want
    to keep them the same as 'amaxs' and 'mdot', then set the list equal to None

    fill: This will change the zero padding for jobnum in labelend.

    iwall: [Boolean] Turns on/off all the disk components and just runs the
    wall. Generally set to 'False'
    Note: This is not looped over (i.e., all the models with this set to 'True'
    will only be walls).

NOTES:
    By default, the standard python wrapper (jobrunner.py) will be used. If you
    want to use a different one, you can give its path to the
    edge.create_runall_py() function called at the end through the parameter
    pyrunner_path.

    'amaxs', 'amaxw', and lamaxb only accept certain values. 'amaxw' accepts the
    same values as 'amaxs'. Here are the possible values:
        amaxs/amaxw: ['0.05', '0.1', '0.25', '0.5', '0.75', '1.0', '1.25',
        '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '10.0',
        '100.0']

        amaxb: ['100','200','300','400',500', '600', '700', '800',' 900',
        '1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000', '9000',
        '10000', '11000', '12000', '13000', '14000', '15000', '16000', '17000',
        '18000', '19000', '20000', '21000', '22000', '23000', '24000', '25000']
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

#Define parameters to feed into file, must be filled with at least 1 value
#Want to check that the values for amaxs and epsilon are possible in the sample job file
mstar   = [1.3] #Mass of the star in Msun
tstar   = [4730] #Temperature of the star in K
rstar   = [1.6] #Radius of the star in solar radii
dist    = [140] #Distance to the star in pc
mdot    = [3.3e-9] #Mass accretion rate in the disk in solar masses per year
mdotstar= [None] #Mass accretion rate at the star. (optional, if None, it will be mdot)

amaxs   = [0.25] #Maximum grain size in the upper layers of the disk NOTE: Only acceptss certain values, see the docstring above
amaxw   = [None] #Maximum grain size at the wall. (optional, if None, it will be amaxs)
amaxb   = [1000] #Maximum grain size in the midplane NOTE: Only accepts certain values
alpha   = [1e-2] # Viscosity in the disk
alphaset= [1e-2] # Turblent viscosity for settling
rdisk   = [200] #Outer radius of the disk

rc      = [2000.] #Critial radius of tapered edge. If 2000, it will not have any effect.
gamma   = [2.0] #Exponent of tapered edge.

temp    = [1400] #Sublimation temperature of the grains
altinh  = [1] #Height of the disk in scale heights. Often better to leave as 1 and scale it later

mui     = [0.5] #Cosine of the inclination of the disk.
tshock  = [8000] #Temperature of the shock. Usually left at 8000K

fracolive = [1.0] #Fraction of olivines
fracpyrox = [0.0] #Fraction of pyroxine
fracforst = [0.0] #Fraction of fosterite
fracent   = [0.0] #Fraction of enstatite

d2g = [0.0065] #Dust to gas mass ratio.

#Gap creator:
#NOTE: these are python lists, so should be written to the jobfile as "[XX]"
#All used deltas need to have the same length as inter_r, so be careful with the combinations.
inter_r = ['[]']
rho_deltas = ['[]']
temp_deltas = ['[]']
epsbig_deltas = ['[]']
eps_deltas = ['[]']

# Non-iterative parameters
# These parameters will affect all the models
fill = 3 #Zero padding for the job numbers. Unless you are running many models (>1000) 3 is standard.

# Switches
imod = False # If True, the disk structure will be modified with the gap creator parameters.
iwall = False #Set this to True if you want ONLY walls. (i.e., turns off the disk code).


#***********************************************
#Unlikely you need to change anything below here.
#***********************************************

#Open up a file and print the parameter names
f = open(gridpath+paramfiletag+'_params.inp', 'w')
f.writelines('JobNum,MSTAR,TSTAR,RSTAR,DISTANCE,MDOT,MDOTSTAR,AMAXS,AMAXW,'+
'AMAXB,ALPHA,ALPHASET,MUI,RDISK,RC,GAMMA,TEMP,ALTINH,D2G,TSHOCK,'+
'AMORPFRAC_OLIVINE,AMORPFRAC_PYROXENE,FORSTERITE_FRAC,ENSTATITE_FRAC,INTER_R,'+
'RHO_DELTAS,TEMP_DELTAS,EPSBIG_DELTAS,EPS_DELTAS,IMOD,IWALLDUST,labelend\n')

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(mstar, tstar, rstar, dist, mdot,
mdotstar, amaxs, amaxw, amaxb, alpha, alphaset, mui, rdisk, rc, gamma, temp,
altinh, d2g, tshock, fracolive, fracpyrox, fracforst, fracent,
inter_r, rho_deltas, temp_deltas, epsbig_deltas, eps_deltas)):

    #Handle the cases of mdotstar and amaxw which are optional parameters
    #NOTE: DO NOT CHANGE THE PARAMETERS BEFORE MDOTSTAR OR THIS WILL BREAK! YOU HAVE BEEN WARNED
    values = list(values)
    if mdotstar == [None]:
        values[5] = values[4]
    if amaxw == [None]:
        values[7] = values[6]
    values.append(imod)
    values.append(iwall)
    values.append(labelend+'_'+str(ind+jobnumstart).zfill(fill))
    f.writelines(str(ind+jobnumstart)+','+str(values).replace(' ','').replace("'","")[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(gridpath+paramfiletag+'_params.inp')
#Throw up a warning if you are making a huge grid
if len(table) > 1000:
    print('WARNING! GRID IS LARGE AND WILL USE SIGNIFICANT LIMITED COMPUTING '+
    'RESOURCES! SPEAK TO A SUPERVISOR BEFORE RUNNING ON THE CLUSTER!!!')
    warning = input('Please enter "I understand" to continue making this grid: ')
    if warning != 'I understand':
        raise ValueError('Grid creation canceled.')
    else:
        print('Continuing grid creation process...')

#Make a run all file for the grid you just created
edge.create_runall_py(jobnumstart, jobnumstart+len(table)-1,
paramfiletag+'_params.inp', clusterpath, outpath = gridpath,
fill = fill)
