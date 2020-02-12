import itertools
import numpy as np
import EDGE as edge
from astropy.io import ascii
import pdb

'''
jobmaker.py

PURPOSE:
    Script that uses job_file_create to produce a file with all the parameters in the grid and creates the job files
    themselves.

HOW TO USE THIS SCRIPT:
    1) Change the gridpath to be where you want the jobfiles to be placed. NOTE: This should also be the location of the jobsample file.

    2) Change the paramfiletag to an insightful name that will help you identify the grid of models you ran three months from now.

    3) Change the parameters in the brackets below to whatever set of parameters you want to run.
       The parameters should be surrounded by []'s and be separated by a commas (A list)

    4) Change the labelname to the name of the object, e.g. 'gmauriga'

    5) Run the script

OPTIONAL PARAMETERS:
    'amaxw' and 'mdotstar': Generally these take the value of 'amaxs' and 'mdot' respectively, but it is possible that you may want to change them.
    To do this, change them like you would any other parameter. If you DO want to keep them the same as 'amaxs' and 'mdot', then set the list equal to None

    fill: This will change the zero padding for the job files.

    iwall: [Boolean] Turns on/off all the disk components and just runs the wall. Generally set to 'False'
           Note: This is not looped over (i.e., all the models with this set to 'True' will only be walls).

NOTES:
    'amaxs', 'amaxw', and lamaxb only accept certain values. 'amaxw' accepts the same values as 'amaxs'
    Here are the possible values:
        amaxs/amaxw: ['0.05', '0.1', '0.25', '0.5', '0.75', '1.0', '1.25',
        '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '10.0',
        '100.0']

        lamaxb: ['100','200','300','400',500', '600', '700', '800',' 900',
        '1mm', '2mm', '3mm', '4mm', '5mm', '6mm', '7mm', '8mm', '9mm', '1cm',
        '1p1cm', '1p2cm', '1p3cm', '1p4cm', '1p5cm', '1p6cm', '1p7cm', '1p8cm',
        '1p9cm', '2cm', '2p1cm', '2p2cm', '2p3cm', '2p4cm', '2p5cm']

MODIFICATION HISTORY
    Jun 16th, 2017 - Updated for mdotstar + amaxw by Connor Robinson
    Apr 29th, 2016 - Written by Connor Robinson
'''

#Where you want the parameter file and the jobfiles to be placed
#Also must be the location of the sample job file
gridpath = '/Users/Connor/Desktop/Research/diad/test/'

#Where you will be running the jobs on the cluster
clusterpath = '/projectnb/bu-disks/connorr/test/'

#Tag that you can add to make the parameter file identifiable for a given run
#Can leave it blank if you don't care.
paramfiletag = 'testgrid'

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

amaxs   = [0.25] #Maximum grain size in the upper layers of the disk NOTE: Only acceptss certain values, see the docstring above
epsilon = [0.1] #Settling parameter
ztran   = [0.1] #height of transition between big and small grains, in hydrostatic scale heights
alpha   = [1e-2] # Viscosity in the disk
rdisk   = [200] #Outer radius of the disk

rc      = [2000.] #Critial radius of tapered edge. If 2000, it will not have any effect.
gamma   = [2.0] #Exponent of tapered edge.

temp    = [1400] #Sublimation temperature of the grains
altinh  = [1] #Height of the disk in scale heights. Often better to leave as 1 and scale it later

lamaxb  = ['1mm'] #Maximum grain size in the midplane NOTE: Only accepts certain values
mui     = [0.5] #Cosine of the inclination of the disk.
tshock  = [8000] #Temperature of the shock. Usually left at 8000K

fracolive = [1.0] #Fraction of olivines
fracpyrox = [0.0] #Fraction of pyroxine
fracforst = [0.0] #Fraction of fosterite
fracent   = [0.0] #Fraction of enstatite

d2g = [0.0065] #Dust to gas mass ratio.

#Gap creator:
imod = False # If True, the disk structure will be modified as chosen with the following parameters.
#NOTE: these are python lists, so should be written to the jobfile as "[XX]"
#All used deltas need to have the same length as inter_r, so be careful with the combinations.
inter_r = ['[]']
rho_deltas = ['[]']
temp_deltas = ['[]']
epsbig_deltas = ['[]']
eps_deltas = ['[]']

#OPTIONAL PARAMETERS:
#If you want them to be the same as their associated parameters (amaxs, mdot) then set them to [None]
#Don't forget the brackets.
mdotstar = [None] #Mass accretion rate at the star
amaxw   = [None] #Maximum grain size at the wall

fill = 3 #Zero padding for the job numbers. Unless you are running many models (>1000) 3 is standard.

iwall = False #Set this to True if you want ONLY walls. (i.e., turns off the disk code).


#***********************************************
#Unlikely you need to change anything below here.
#***********************************************

#Open up a file and print the parameter names
f = open(gridpath+paramfiletag+'job_params.txt', 'w')
f.writelines('Job Number, amaxs, amaxw, epsilon, ztran, mstar, tstar, rstar, \
dist, mdot, mdotstar, tshock, alpha, mui, rdisk, rc, gamma, temp, altinh, \
fracolive, fracpyrox, fracforst, fracent, lamaxb, d2g, inter_r, rho_deltas, \
temp_deltas, epsbig_deltas, eps_deltas \n')

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(amaxs, amaxw, epsilon, ztran,
mstar, tstar, rstar, dist, mdot, mdotstar, tshock, alpha, mui, rdisk, rc,
gamma, temp, altinh, fracolive, fracpyrox, fracforst, fracent, lamaxb, d2g,
inter_r, rho_deltas, temp_deltas, epsbig_deltas, eps_deltas)):

    #Handle the cases of mdotstar and amaxw which are optional parameters
    #NOTE: DO NOT CHANGE THE PARAMETERS BEFORE MDOTSTAR OR THIS WILL BREAK! YOU HAVE BEEN WARNED
    values = np.array(values)
    if mdotstar == [None]:
        values[9] = values[8]
    if amaxw == [None]:
        values[1] = values[0]
    values = tuple(values)

    f.writelines(str(ind+jobnumstart)+', '+str(values)[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(gridpath+paramfiletag+'job_params.txt')

#Throw up a warning if you are making a huge grid
if len(table) > 1000:
    print('WARNING! GRID IS LARGE AND WILL USE SIGNIFICANT LIMITED COMPUTING RESOURCES! SPEAK TO A SUPERVISOR BEFORE RUNNING ON THE CLUSTER!!!')
    warning = input('Please enter "I understand" to continue making this grid: ')

    if warning != 'I understand':
        raise ValueError('Job creation canceled.')
    else:
        print('Continuing job creation process...')

#Create the jobfiles using edge.job_file_create
for i in range(len(table)):
    label = labelend+'_'+str(i+jobnumstart).zfill(fill)

    # Remove the apostrophes
    if "'" in table['lamaxb'][i]:
        table['lamaxb'][i] = table['lamaxb'][i].split("'")[1]
    if "'" in table['inter_r'][i]:
        table['inter_r'][i] = table['inter_r'][i].split("'")[1]
    if "'" in table['rho_deltas'][i]:
        table['rho_deltas'][i] = table['rho_deltas'][i].split("'")[1]
    if "'" in table['temp_deltas'][i]:
        table['temp_deltas'][i] = table['temp_deltas'][i].split("'")[1]
    if "'" in table['epsbig_deltas'][i]:
        table['epsbig_deltas'][i] = table['epsbig_deltas'][i].split("'")[1]
    if "'" in table['eps_deltas'][i]:
        table['eps_deltas'][i] = table['eps_deltas'][i].split("'")[1]

    edge.job_file_create(i+jobnumstart, gridpath, \
    amaxs     = table['amaxs'][i],\
    eps       = table['epsilon'][i],\
    ztran     = table['ztran'][i],\
    mstar     = table['mstar'][i], \
    tstar     = table['tstar'][i], \
    rstar     = table['rstar'][i], \
    dist      = table['dist'][i], \
    mdot      = table['mdot'][i], \
    mdotstar  = table['mdotstar'][i], \
    tshock    = table['tshock'][i], \
    alpha     = table['alpha'][i], \
    mui       = table['mui'][i], \
    rdisk     = table['rdisk'][i], \
    rc        = table['rc'][i], \
    gamma     = table['gamma'][i], \
    temp      = table['temp'][i], \
    altinh    = table['altinh'][i],\
    fracolive = table['fracolive'][i], \
    fracpyrox = table['fracpyrox'][i], \
    fracforst = table['fracforst'][i], \
    fracent   = table['fracent'][i], \
    amaxw     = table['amaxw'][i],\
    lamaxb    = table['lamaxb'][i],\
    d2g       = table['d2g'][i],\
    inter_r   = table['inter_r'][i],\
    rho_deltas= table['rho_deltas'][i],\
    temp_deltas= table['temp_deltas'][i],\
    epsbig_deltas= table['epsbig_deltas'][i],\
    eps_deltas= table['eps_deltas'][i],\
    fill      = fill,\
    iwall     = iwall,\
    imod      = imod,\
    labelend  = label)

#Make a run all file for the jobs you just created
edge.create_runall(jobnumstart, jobnumstart+len(table)-1, clusterpath, optthin = False, outpath = gridpath, fill = fill)
