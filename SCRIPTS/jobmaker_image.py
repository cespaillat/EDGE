import itertools
import numpy as np
import EDGE as edge
from astropy.io import ascii
import pdb

'''
jobmaker_image.py

PURPOSE:
    Script that uses job_file_create to produce the job files to run the IMAGE code.

    NOTE: In order to run an image, you first need to run the SED code with the same parameters in the same directory.
    The SED code will calculate the structure of the disk, together with other necessary files,that are then used by
    the image code to compute an image at a particular wavelength.

HOW TO USE THIS SCRIPT:
    1) Change the gridpath to be where you want the jobfiles to be placed. NOTE: This should also be the location of the job_image (sample) file.

    2) Change the the wavelength and image (thin or thick) parameters.

    3) Change the parameters in the brackets below to the parameters used in the SED model.

    4) Change the labelend to be the same one as the SED model for which you want to run an image.

    5) Run the script

NOTES:
    - 'amaxs', 'amaxw', and lamaxb only accept certain values. 'amaxw' accepts the same values as 'amaxs'
    Here are the possible values:
        amaxs/amaxw: ['0.05', '0.1', '0.25', '0.5', '0.75', '1.0', '1.25',
        '1.5', '1.75', '2.0', '2.25', '2.5', '3.0', '4.0', '5.0', '10.0',
        '100.0']

        lamaxb: ['100','200','300','400',500', '600', '700', '800',' 900',
        '1mm', '2mm', '3mm', '4mm', '5mm', '6mm', '7mm', '8mm', '9mm', '1cm',
        '1p1cm', '1p2cm', '1p3cm', '1p4cm', '1p5cm', '1p6cm', '1p7cm', '1p8cm',
        '1p9cm', '2cm', '2p1cm', '2p2cm', '2p3cm', '2p4cm', '2p5cm']
    - 'image' can be set to either 'thin' or 'thick', depending on whether the disk will be optically thin or thick
    at the wavelength of the image.
    - 'mdotstar' and 'tshock' are not yet supported by the image code.
    - 'frac...' are not needed, since the code will read the silicate opacities calculated by the SED.

'''

#Where you want the parameter file and the jobfiles to be placed
#Also must be the location of the sample job file
gridpath = '/Users/Connor/Desktop/Research/diad/test/'

#Where you will be running the jobs on the cluster
clusterpath = '/projectnb/bu-disks/connorr/test/'

labelend = 'test_001'

#What number to start counting from (for the names of the jobfiles), must be an integer
jobnumstart = 1

######################
## IMAGE parameters ##
######################

wavelengths = [1300] # wavelengths at which the image will be calculated, in microns

# Type of image. Can be thick or thin, depending on whether the
# disk will be optically thin or optically thick at this wavelength
imagetype = 'thin'

######################
### DISK parameters ##
######################
#Define parameters to feed into file, must be the same values as in the SED model

mstar   = 1.3 #Mass of the star in Msun
tstar   = 4730 #Temperature of the star in K
rstar   = 1.6 #Radius of the star in solar radii
dist    = 140 #Distance to the star in pc
mdot    = 3.3e-9 #Mass accretion rate in the disk in solar masses per year

amaxs   = 0.25 #Maximum grain size in the upper layers of the disk NOTE: Only acceptss certain values, see the docstring above
amaxw   = amaxs #Maximum grain size in the wall
epsilon = 0.1 #Settling parameter
alpha   = 1e-2 # Viscosity in the disk
rdisk   = 30 #Outer radius of the disk

temp    = 1400 #Sublimation temperature of the grains
altinh  = 1 #Height of the disk in scale heights. Often better to leave as 1 and scale it later

lamaxb  = '500' #Maximum grain size in the midplane NOTE: Only accepts certain values
mui     = 0.5 #Cosine of the inclination of the disk.

d2g = 0.0065 #Dust to gas mass ratio.

fill = 3 #Zero padding for the job numbers. Unless you are running many models (>1000) 3 is standard.

#***********************************************
#Unlikely you need to change anything below here.
#***********************************************

#Create the jobfiles using edge.job_file_create
for i,wl in enumerate(wavelengths):
    edge.job_file_create(i+jobnumstart, gridpath, \
    image     = True,\
    wavelength = wl,\
    imagetype = imagetype,\
    amaxs     = amaxs,\
    eps       = epsilon,\
    mstar     = mstar, \
    tstar     = tstar, \
    rstar     = rstar, \
    dist      = dist, \
    mdot      = mdot, \
    alpha     = alpha, \
    mui       = mui, \
    rdisk     = rdisk, \
    temp      = temp, \
    altinh    = altinh,\
    amaxw     = amaxw,\
    lamaxb    = lamaxb,\
    d2g       = d2g,\
    fill      = fill,\
    labelend  = labelend)

#Make a run all file for the jobs you just created
edge.create_runall(jobnumstart, jobnumstart+len(wavelengths)-1, clusterpath, image = True, optthin = False, outpath = gridpath, fill = fill)
