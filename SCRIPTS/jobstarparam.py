#!/usr/bin/env python
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import util
import os
import EDGE as edge
import starparam

#Set up path to where EDGE/MODULES resides
modulespath = os.path.dirname(os.path.realpath(util.__file__)) + '/'
#Set up path to where EDGE/COMMON resides
commonpath = os.path.realpath(modulespath + '../COMMON') + '/'
#Set up path to where you'd like to put your output files
outpath = './'

# interactive Av determination? if False, it just uses input Av
inter = True

# object name for labeling output file
obj='gmaur'

#spectral type; should be a whole number
sptin='K5'

# uncomment extinction law to use, default is mcclure
#law='mathis'
#law='HD29647'
#law='CCM89'
law='mcclure'

# uncomment which color table to use
table='kh95'
#table='pm13'

# uncomment which isochrones to use
#isochrone = 'baraffe'
isochrone = 'siess'

# uncomment to turn on/off HR diagram display
HR = True

# uncomment Rv to use
r=3.1
#r=5.0

# Av
avin=0.8

# distance in pc
distance=160

# If you have an obs fits file created with EDGE, you can
# input the photometry using it. You might need to change
# some code below to select the correct photometry with
# the names you gave them
input_obs = True
if input_obs:
    c = 2.99793e10
    source_obs = edge.loadObs(obj,outpath)
    xu = util.convertJy_to_Mag(source_obs.photometry['UBVRI']['lFl'][0]*
    source_obs.photometry['UBVRI']['wl'][0]*1e-4/c*1e23,'U')
    xb = util.convertJy_to_Mag(source_obs.photometry['UBVRI']['lFl'][1]*
    source_obs.photometry['UBVRI']['wl'][1]*1e-4/c*1e23,'B')
    xv = util.convertJy_to_Mag(source_obs.photometry['UBVRI']['lFl'][2]*
    source_obs.photometry['UBVRI']['wl'][2]*1e-4/c*1e23,'V')
    xr = util.convertJy_to_Mag(source_obs.photometry['UBVRI']['lFl'][3]*
    source_obs.photometry['UBVRI']['wl'][3]*1e-4/c*1e23,'R')
    xi = util.convertJy_to_Mag(source_obs.photometry['UBVRI']['lFl'][4]*
    source_obs.photometry['UBVRI']['wl'][4]*1e-4/c*1e23,'I')
    xj = util.convertJy_to_Mag(source_obs.photometry['2MASS']['lFl'][0]*
    source_obs.photometry['2MASS']['wl'][0]*1e-4/c*1e23,'J')
    xh = util.convertJy_to_Mag(source_obs.photometry['2MASS']['lFl'][1]*
    source_obs.photometry['2MASS']['wl'][1]*1e-4/c*1e23,'H')
    xk = util.convertJy_to_Mag(source_obs.photometry['2MASS']['lFl'][2]*
    source_obs.photometry['2MASS']['wl'][2]*1e-4/c*1e23,'K')
    xl = np.nan
    xm = np.nan
    irac1 = util.convertJy_to_Mag(source_obs.photometry['IRAC']['lFl'][0]*
    source_obs.photometry['IRAC']['wl'][0]*1e-4/c*1e23,'IRAC3.6')
    irac2 = util.convertJy_to_Mag(source_obs.photometry['IRAC']['lFl'][1]*
    source_obs.photometry['IRAC']['wl'][1]*1e-4/c*1e23,'IRAC4.5')
    irac3 = util.convertJy_to_Mag(source_obs.photometry['IRAC']['lFl'][2]*
    source_obs.photometry['IRAC']['wl'][2]*1e-4/c*1e23,'IRAC5.8')
    irac4 = util.convertJy_to_Mag(source_obs.photometry['IRAC']['lFl'][3]*
    source_obs.photometry['IRAC']['wl'][3]*1e-4/c*1e23,'IRAC8.0')
    mips1 = util.convertJy_to_Mag(source_obs.photometry['MIPS']['lFl'][0]*
    source_obs.photometry['MIPS']['wl'][0]*1e-4/c*1e23,'MIPS24')

else: #If not, input it by hand
    # input photometry in magnitudes, when missing a value np.nan
    # need to list at least J-band ('xj') for SpT less than ~G
    # for earlier type stars need to list at least V-band ('xv')
    # to get Mdot need U-band ('xu') in addition to above
    # put 99.00 for magnitudes with no value
    xu=13.6
    xb=12.5
    xv=11.4
    xr=10.4
    xi=9.74
    xj=8.90
    xh=8.47
    xk=8.58
    xl=np.nan
    xm=np.nan
    irac1=8.04
    irac2=7.88
    irac3=7.68
    irac4=7.07
    mips1=2.48

# what to call output file
outputfile=outpath+'starparam.'+obj+'.'+law+'.rv'+str(r)+'.av'+str(avin)+'.'+table

# FOR TEMPLATE PHOTOSPHERE

# do you want an output template photosphere (useful when fitting for Av)
calcphot = True

# what to call output photosphere file
photfile=outpath+'photosphere.'+obj+'.'+law+'.rv'+str(r)+'.av'+str(avin)+'.'+table

# wavelengths over which to interpolate template photosphere
# use 'wlfile_standard.ent' if you're using the photosphere in SED models
photfilewl=commonpath+'wavelengths/'+'longitudes_4testruns_shorter.ent'
#photfilewl=commonpath+'wavelengths/'+'wlfile_standard.ent'

#
starparam.starparam(obj, sptin, avin, distance, law, table, isochrone, HR,
calcphot, inter, r,
xu, xb, xv, xr, xi, xj, xh, xk, xl, xm, irac1, irac2, irac3, irac4, mips1,
outpath, outputfile, photfile, commonpath, photfilewl)
