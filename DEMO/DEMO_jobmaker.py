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
    'amaxs', 'amaxw' and 'epsilon' only accept certain values. 'amaxw' accepts the same values as 'amaxs'
    Here are the possible values:
        amaxs: [0.1, 0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 10, 100]
        epsilon: [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0]

MODIFICATION HISTORY
    Jun 16th, 2017 - Updated for mdotstar + amaxw by Connor Robinson
    Apr 29th, 2016 - Written by Connor Robinson
'''

#Where you want the parameter file and the jobfiles to be placed
#Also must be the location of the sample job file
gridpath = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/models/'

#Where you will be running the jobs on the cluster
clusterpath = '/projectnb/bu-disks/connorr/test/'

#Tag that you can add to make the parameter file identifiable for a given run
#Can leave it blank if you don't care. 
paramfiletag = 'DEMO_imlup_'

#No need to add an underscore/jobnumber, the script will do that for you.
labelend = 'imlup'

#What number to start counting from, must be an integer 
jobnumstart = 1

#Define parameters to feed into file, must be filled with at least 1 value
#Want to check that the values for amaxs and epsilon are possible in the sample job file

mstar   = [1.0] #Mass of the star in Msun
tstar   = [3850] #Temperature of the star in K
rstar   = [3.0] #Radius of the star in solar radii
dist    = [161] #Distance to the star in pc
mdot    = [9e-9] #Mass accretion rate in the disk in solar masses per year

amaxs   = [3.0] #Maximum grain size in the upper layers of the disk NOTE: Only acceptss certain values, see the docstring above
epsilon = [0.001] #Settling parameter NOTE: Only accepts certain values, see the docstring above
alpha   = [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005, \
           0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01] # Viscosity in the disk 
rdisk   = [300] #Outer radius of the disk

temp    = [500] #Sublimation temperature of the grains
altinh  = [1] #Height of the disk in scale heights. Often better to leave as 1 and scale it later

lamaxb  = ['1mm'] #Maximum grain size in the midplane
mui     = [0.66913] #Cosine of the inclination of the disk. 
tshock  = [8000] #Temperature of the shock. Usually left at 8000K

fracolive = [1] #Fraction of olivines 
fracpyrox = [0] #Fraction of pyroxine
fracforst = [0] #Fraction of fosterite
fracent   = [0] #Fraction of enstatite

d2g = [0.0065] #Dust to gas mass ratio.

#OPTIONAL PARAMETERS:
#If you want them to be the same as their associated parameters (amaxs, mdot) then set them to [None]
#Don't forget the brackets.
mdotstar = [None] #Mass accretion rate at the star
amaxw    = [None] #Maximum grain size at the wall

fill = 3 #Zero padding for the job numbers. Unless you are running many models (>1000) 3 is standard.

iwall = False #Set this to True if you want ONLY walls. (i.e., turns off the disk code).

#***********************************************
#Unlikly you need to change anything below here.
#***********************************************

#Open up a file and print the parameter names
f = open(gridpath+paramfiletag+'job_params.txt', 'w') 
f.writelines('Job Number, amaxs, amaxw, epsilon, mstar, tstar, rstar, dist, mdot, mdotstar, tshock, alpha, mui, rdisk, temp, altinh, fracolive, fracpyrox, fracforst, fracent, lamaxb, d2g \n') 

#Write each iteration as a row in the table
for ind, values in enumerate(itertools.product(amaxs, amaxw, epsilon, mstar, tstar, rstar, dist, mdot, mdotstar, tshock, alpha, mui, rdisk, temp, altinh, fracolive, fracpyrox, fracforst, fracent, lamaxb, d2g)):
    
    #Handle the cases of mdotstar and amaxw which are optional parameters
    #NOTE: DO NOT CHANGE THE PARAMETERS BEFORE MDOTSTAR OR THIS WILL BREAK! YOU HAVE BEEN WARNED
    values = np.array(values)
    if mdotstar == [None]:
        values[8] = values[7] 
    if amaxw == [None]:
        values[1] = values[0]
    values = tuple(values)
    
    f.writelines(str(ind+jobnumstart)+', '+str(values)[1:-1]+ '\n')
f.close()

#Open up the table
table = ascii.read(gridpath+paramfiletag+'job_params.txt') 

#Create the jobfiles using edge.job_file_create
for i in range(len(table)):
    label = labelend+'_'+str(i+jobnumstart).zfill(fill)
    
    edge.job_file_create(i+jobnumstart, gridpath, \
    amaxs     = table['amaxs'][i],\
    epsilon   = table['epsilon'][i],\
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
    temp      = table['temp'][i], \
    altinh    = table['altinh'][i],\
    fracolive = table['fracolive'][i], \
    fracpyrox = table['fracpyrox'][i], \
    fracforst = table['fracforst'][i], \
    fracent   = table['fracent'][i], \
    amaxw     = table['amaxw'][i],\
    lamaxb    = table['lamaxb'][i].split("'")[1],\
    d2g       = table['d2g'][i],\
    fill      = fill,\
    iwall     = iwall,\
    labelend  = label)

#Make a run all file for the jobs you just created
edge.create_runall(jobnumstart, jobnumstart+len(table)-1, clusterpath, optthin = False, outpath = gridpath, fill = fill)
