#!/usr/bin/env python

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import util 

#Set up path to where EDGE/MODULES resides
modulespath = '/Users/ccespa/Desktop/Dropbox/analysis/tools/EDGE/MODULES/'
#Set up path to where EDGE/COMMON resides
commonpath = '/Users/ccespa/Desktop/Dropbox/analysis/tools/EDGE/COMMON/'
#Set up path to where you'd like to put your output files
outpath = '/Users/ccespa/Desktop/Dropbox/analysis/tests/starparam/'

# object name for labeling output file
obj='gmaur'

#spectral type; should be a whole number
sptin='K5' 

# uncomment extinction law to use
law='mathis'
#law='HD29647'
#law='CCM89'
#law='mcclure'

# uncomment which color table to use
table='kh95'
#table='pm13'

# uncomment Rv to use
r=3.1 
#r=5.0

# Av 
avin=0.8 

# distance in pc
distance=140

# input photometry in magnitudes, when missing a value '99.00'
# need to list at least J-band ('xj') for SpT less than ~G
# for earlier type stars need to list at least V-band ('xv')
# to get Mdot need U-band ('xu') in addition to above
xu=13.6
xb=12.5
xv=11.4
xr=10.4
xi=9.74
xj=8.90
xh=8.47
xk=8.58
xl=99.00
xm=99.00
irac1=8.04
irac2=7.88
irac3=7.68
irac4=7.07
mips1=2.48

# what to call output file
outputfile=outpath+'starparam.'+obj+'.'+law+'.rv'+str(r)+'.av'+str(avin)+'.'+table

# FOR TEMPLATE PHOTOSPHERE

# do you want an output template photosphere (useful when fitting for Av)
calcphot = 'yes'
#calcphot = 'no'

# what to call output photosphere file
photfile=outpath+'photosphere.'+obj+'.'+law+'.rv'+str(r)+'.av'+str(avin)+'.'+table

# wavelengths over which to interpolate template photosphere
photfilewl=commonpath+'wavelengths/'+'longitudes_4testruns_shorter.ent'

#exec("starparam.py")
exec(compile(open(modulespath+"starparam.py").read(), modulespath+"starparam.py", 'exec'))