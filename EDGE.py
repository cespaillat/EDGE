#!/usr/bin/env python
# Created by Dan Feldman and Connor Robinson for analyzing data from Espaillat Group research models.
# Last updated: 12/14/16 by Connor, Dan and Kike!


#---------------------------------------------IMPORT RELEVANT MODULES--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
#from astropy.io import ascii
from astropy.io import fits
import scipy.interpolate as sinterp
#from matplotlib.backends.backend_pdf import PdfPages
import os
import itertools
import math
import _pickle as cPickle
import pdb
import copy
#----------------------------------------------PLOTTING PARAMETERS-----------------------------------------------
# Regularizes the plotting parameters like tick sizes, legends, etc.
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')
#plt.rc('text', usetex=True)
plt.rc('legend', fontsize=10)
plt.rc('axes', labelsize=15)
plt.rc('figure', autolayout=True)

#-----------------------------------------------------PATHS------------------------------------------------------
# Folders where model output data and observational data can be found:
edgepath        = os.path.dirname(os.path.realpath(__file__))+'/'
commonpath     = edgepath+'COMMON/'
datapath        = '/Users/Connor/Desktop/Research/iceline/data/'
figurepath      = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/Models/Full_CVSO_Grid/CVSO58_sil/'
shockpath       = '/Users/danfeldman/Orion_Research/Orion_Research/CVSO_4Objs/ob1bspectra/'

#---------------------------------------------INDEPENDENT FUNCTIONS----------------------------------------------
# A function is considered independent if it does not reference any other function or class in this module.

def keyErrHandle(func):
    """
    A decorator to allow methods and functions to have key errors, and to print the failed key.
    """

    def handler(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyError as badKey:
            print('Key error was encountered. The missing key is: ' + str(badKey))
            return 0
        else:
            return 1
    return handler

def filelist(path):
    """
    Returns a list of files in a directory. Pops out hidden values.

    INPUTS
    path: The directory path from which we wish to grab a file list.

    OUTPUT
    flist: The file list. It should be devoid of hidden files.
    """

    flist       = os.listdir(path)                              # Full list of files
    hid         = []
    for f in flist:
        if f.startswith('.'):                                   # If hidden file, tag it
            hid.append(flist.index(f))
    for index, val in enumerate(hid):                           # Pop out tagged entries
        flist.pop(val - index)
    return flist

def deci_to_time(ra=None, dec=None):
    """
    Converts decimal values of ra and dec into arc time coordinates.

    INPUTS
    ra: The float value of right ascension.
    dec: The float value of declination.

    OUTPUTS
    new_ra: The converted RA. If no ra supplied, returns -1
    new_dec: The converted dec. If no dec supplied, returns -1
    """

    new_ra  = -1
    new_dec = -1

    if ra is not None:
        if type(ra) != float:
            raise ValueError('DECI_TO_TIME: RA is not a float. Cannot convert.')

        # First, we find the number of hours:
        hours    = ra / 15.0
        hoursInt = int(hours)
        hours    = hours - hoursInt

        # Next, we want minutes:
        minutes  = hours * 60.0
        minInt   = int(minutes)
        minutes  = minutes - minInt

        # Lastly, seconds:
        seconds  = minutes * 60.0
        new_ra   = '{0:02d} {1:02d} {2:.2f}'.format(hoursInt, minInt, seconds)

    if dec is not None:
        if type(dec) != float:
            raise ValueError('DECI_TO_TIME: Dec is not a float. Cannot convert.')

        # For dec, have to check and store the sign:
        if dec < 0.0:
            sign = '-'
        else:
            sign = '+'
        dec      = abs(dec)

        # First, we find the number of degrees:
        degInt   = int(dec)
        deg      = dec - degInt

        # Next, we want minutes:
        minutes  = deg * 60.0
        minInt   = int(minutes)
        minutes  = minutes - minInt

        # Lastly, seconds:
        seconds  = minutes * 60.0
        new_dec  = '{0:s}{1:02d} {2:02d} {3:.2f}'.format(sign, degInt, minInt, seconds)

    return new_ra, new_dec

def time_to_deci(ra='', dec=''):
    """
    Converts arc time coordinates of ra and dec into degree values. Adapted from BDNYC
    code written by Joe Filippazzo.

    INPUTS
    ra: The string coordinates of right ascension.
    dec: The string coordinates of declination.

    OUTPUTS
    RA: The converted RA.
    DEC: The converted dec.
    """

    RA, DEC, rs, ds = '', '', 1, 1
    if dec:
        D, M, S     = [float(i) for i in dec.split()]
        if str(D)[0] == '-':
            ds, D   = -1, abs(D)
        deg = D + (M/60) + (S/3600)
        DEC = '{0}'.format(deg*ds)

    if ra:
        H, M, S     = [float(i) for i in ra.split()]
        if str(H)[0] == '-':
            rs, H   = -1, abs(H)
        deg = (H*15) + (M/4) + (S/240)
        RA  = '{0}'.format(deg*rs)

    if ra and dec:
        return (RA, DEC)
    else:
        return RA or DEC

def calcAngularDist(coords1, coords2):
    """
    Calculates the angular distance between two points on the sky. Inputs should be in degrees.

    INPUTS
    coords1: A list containing the RA and Dec for the first position. Should be [RA, Dec]
    coords2: A list containing the RA and Dec for the second position. Also [RA, Dec]

    OUTPUT
    angDist: The angular distance in degrees.
    """

    deltaRA = float(coords1[0]) - float(coords2[0])
    deltaDEC= float(coords1[1]) - float(coords2[1])
    decRads = float(coords1[0])*np.pi/180.              # Dec in radians
    angDist = math.sqrt((deltaRA*math.cos(decRads)**2.0) + (deltaDEC**2.0))

    return angDist



def linearInterp(x0, x1, x2, y1, y2, y1err, y2err):
    """
    Linearly interpolates between two values assuming the y values have errors.

    INPUTS
    x0: The x value of where you wish to interpolate.
    x1: The lower x value bound.
    x2: The upper x value bound.
    y1: The y value corresponding to x1.
    y2: The y value corresponding to x2.
    y1err: The error in the y1 value.
    y2err: The error in the y2 value.

    OUTPUTS
    y0: The interpolated y value corresponding to x0.
    yerr: The error in y0.
    """

    y0   = y1 + (y2-y1) * ((x0-x1)/(x2-x1))
    yerr = math.sqrt(2*(y1err**2) + (y2err**2))

    return y0, yerr

def convertFreq(value):
    """
    Convert a frequency value in s-1 to wavelength in microns. Should also work with arrays.

    INPUTS
    value: A frequency value or array of frequency values in s-1 units.

    OUTPUT
    wl: The wavelength or array of wavelength values in microns.
    """

    c_microns   = 2.997924e14                                   # Speed of light in microns
    wl          = c_microns / value

    return wl

def convertJy(value, wavelength):
    """
    Convert a flux in Janskys to erg s-1 cm-2. Should also work with flux/wl arrays of same size.

    INPUTS
    value: A flux value in the units of Jy.
    wavelength: The corresponding wavelength value in microns (or perhaps a central wavelength).

    OUTPUT
    flux: The flux value in units of erg s-1 cm-2.
    """

    c_microns   = 2.997924e14                                   # Speed of light in microns
    flux        = value * 1e-23 * (c_microns / wavelength)      # lamda*F_lambda or nu*F_nu

    return flux

def convertMag(value, band, jy='False', getwl='False'):
    """
    Converts a magnitude into a flux in erg s-1 cm-2. To use this for an array, use np.vectorize().
    Currently handles:
        UBVRI
        JHK
        LMNQ
        griz
        MIPS(24,70,160)
        IRAC (3.6,4.5,5.8,8.0)
        W1-W4 (WISE)

    References: http://people.physics.tamu.edu/lmacri/astro603/lectures/astro603_lect01.pdf
                http://casa.colorado.edu/~ginsbura/filtersets.htm
                http://www.astro.utoronto.ca/~patton/astro/mags.html
                http://ircamera.as.arizona.edu/astr_250/Lectures/Lecture_13.htm

    INPUTS
    value: A magnitude value (units of mag).
    band: The band corresponding to the magnitude value.
    jy: Boolean -- If False, will use convertJy() to convert the flux into erg s-1 cm-2. If True, will
                   leave the output value in Jy.

    OUTPUTS
    flux: The flux value in erg s-1 cm-2.
    fluxJ: The flux value in Jy.

    MODIFICATIONS BY CONNOR:
        Added ability to return the wavelength alongside the flux

    """

    # First convert to Janskys:
    if band.upper()     == 'U':
        fluxJ       = 1810. * (10.0**(value / -2.5))
        wavelength  = 0.367                                     # In Microns
    elif band.upper()   == 'B':
        fluxJ       = 4260. * (10.0**(value / -2.5))
        wavelength  = 0.436
    elif band.upper()   == 'V':
        fluxJ       = 3640. * (10.0**(value / -2.5))
        wavelength  = 0.545
    elif band.upper()   == 'R':
        fluxJ       = 3080. * (10.0**(value / -2.5))
        wavelength  = 0.638
    elif band.upper()   == 'I':
        fluxJ       = 2550. * (10.0**(value / -2.5))
        wavelength  = 0.797
    elif band.upper()   == 'J':
        fluxJ       = 1600. * (10.0**(value / -2.5))
        wavelength  = 1.220
    elif band.upper()   == 'H':
        fluxJ       = 1080. * (10.0**(value / -2.5))
        wavelength  = 1.630
    elif band.upper()   == 'K':
        fluxJ       = 670. * (10.0**(value / -2.5))
        wavelength  = 2.190
    elif band.upper()   == 'L':
        fluxJ       = 281. * (10.0**(value / -2.5))
        wavelength  = 3.450
    elif band.upper()   == 'M':
        fluxJ       = 154. * (10.0**(value / -2.5))
        wavelength  = 4.750
    elif band.upper()   == 'N':
        fluxJ       = 37. * (10.0**(value / -2.5))
        wavelength  = 10.10
    elif band.upper()   == 'Q':
        fluxJ       = 10. * (10.0**(value / -2.5))
        wavelength  = 20.00
    elif band.upper()   == 'SDSSG':
        fluxJ       = 3730. * (10.0**(value / -2.5))
        wavelength  = 0.4686
    elif band.upper()   == 'SDSSR':
        fluxJ       = 4490. * (10.0**(value / -2.5))
        wavelength  = 0.6165
    elif band.upper()   == 'SDSSI':
        fluxJ       = 4760. * (10.0**(value / -2.5))
        wavelength  = 0.7481
    elif band.upper()   == 'SDSSZ':
        fluxJ       = 4810. * (10.0**(value / -2.5))
        wavelength  = 0.8931
    elif band.upper()   == 'MIPS24':
        fluxJ       = 7.17 * (10.0**(value / -2.5))
        wavelength  = 23.675
    elif band.upper()   == 'MIPS70':
        fluxJ       = 0.778 * (10.0**(value / -2.5))
        wavelength  = 71.42
    elif band.upper()   == 'MIPS160':
        fluxJ       = 0.16 * (10.0**(value / -2.5))
        wavelength  = 155.9
    elif band.upper()   == 'IRAC3.6':
        fluxJ       = 280.9 * (10.0**(value / -2.5))
        wavelength  = 3.60
    elif band.upper()   == 'IRAC4.5':
        fluxJ       = 179.7 * (10.0**(value / -2.5))
        wavelength  = 4.50
    elif band.upper()   == 'IRAC5.8':
        fluxJ       = 115. * (10.0**(value / -2.5))
        wavelength  = 5.80
    elif band.upper()   == 'IRAC8.0':
        fluxJ       = 64.13 * (10.0**(value / -2.5))
        wavelength  = 8.0
    elif band.upper()   == 'W1':
        fluxJ       = 309.5 * (10.0**(value / -2.5))
        wavelength  = 3.35
    elif band.upper()   == 'W2':
        fluxJ       = 171.8 * (10.0**(value / -2.5))
        wavelength  = 4.60
    elif band.upper()   == 'W3':
        fluxJ       = 31.67 * (10.0**(value / -2.5))
        wavelength  = 11.56
    elif band.upper()   == 'W4':
        fluxJ       = 8.36 * (10.0**(value / -2.5))
        wavelength  = 22.09
    elif band.upper() == 'GAIAG':
        fluxJ       = 3488 * (10.0**(value / -2.5))
        wavelength = .550


    else:
        raise ValueError('CONVERTMAG: Unknown Band given. Cannot convert.')

    if jy == 'False':
        # Next, convert to flux from Janskys:
        flux        = convertJy(fluxJ, wavelength)  # Ok, so maybe this is a dependent function. Shhhhhhh! :)

        if getwl == True:
            return flux, wavelength
        else:
            return flux

    if getwl == True:
        return fluxJ, wavelength
    else:
        return fluxJ

def convertMagErr(flux, magerr):
    """
    Converts magnitude errors into flux errors

    INPUT:
        flux: Flux values in any flux units (Doensn't matter which type, based on fractional uncertainty)
        magerr: Error in magnitudes

    OUTPUT:
        fluxerr: Error in the flux units

    """

    fluxerr = np.abs(flux*(10**(-magerr/2.5) - 1))
    return fluxerr


def convertSptype(spT):
    """
    Converts a spectral type into its numerical equivalent, based on Alice Perez's conversion table.

    INPUT
    spT: The spectral type. Examples include 'A4', 'F3.5', and 'M2.1'. Must be a string.

    OUTPUT
    spT_float: The spectral type as a float value. See the README file at
               https://github.com/yumiry/Teff_Lum for more details on the conversion.
    """

    if type(spT) != str:
        raise ValueError('CONVERTSPTYPE: Spectral type must be a string!')

    # Pull out the numerical value in spT, e.g., the 5 in 'M5':
    try:
        sub_val = float(spT[1:])
    except ValueError:
        raise ValueError('CONVERTSPTYPE: Spectral type not in correct format! Fix the numerical part.')

    # Now, use the first value (e.g., M in 'M5') and the above numerical value to convert to float:
    if spT[0] == 'B':
        spT_float = 20.0 + sub_val
    elif spT[0] == 'A':
        spT_float = 30.0 + sub_val
    elif spT[0] == 'F':
        spT_float = 40.0 + sub_val
    elif spT[0] == 'G':
        spT_float = 50.0 + sub_val
    elif spT[0] == 'K':
        spT_float = 60.0 + sub_val
        if sub_val >= 8.0:
            print('WARNING: Spectral type is greater than K7 but less than M0...not physical.')
    elif spT[0] == 'M':
        spT_float = 68.0 + sub_val
    else:
        raise ValueError('CONVERTSPTYPE: Spectral type not in correct format! Fix the spectral class.')

    return spT_float

def diskMassCalc(lFl, wl, temp, dist):
    """

    THIS IS A TEST FUNCTION!

    Calculates the disk mass based on a sub-mm flux value. Needs to be in Rayleigh-Jeans
    regime or else it doesn't work. This equation assumes implicity that the gas-to-dust
    ratio is 100. NOTE: THIS IS UNTESTED FOR ACCURACY.

    INPUTS
    lFl: The flux value at the given wavelength, in units of erg s-1 cm-2
    wl: The wavelength of the band. It needs to be sufficiently in Rayleigh-Jeans regime. This
        should be given in microns.
    temp: The temperature of the dust in Kelvin.
    dist: The distance to your object in parsecs.

    OUTPUT
    dmass: The disk mass in solar masses.
    """

    print('WARNING! THIS IS UNTESTED!')

    # Define the constants and convert to CGS units:
    K       = 1.381e-16             # Boltzmann constant in cgs
    C       = 3.0e10                # Speed of light in cgs
    NUM     = 0.5e13                # Extra constant needed for equation in units of Hz
    SOLMASS = 1.989e33              # Solar mass in cgs
    wl_cgs  = wl / 1e4              # Wavelength conversion from microns to cm
    d_cgs   = dist * 3.09e18        # Distance to object in cgs

    # Calculate the disk mass using equation from Williams & Cieza 2011:
    dmass   = (NUM * lFl * (d_cgs**2.0) * (wl_cgs**4.0)) / (K * temp * (C**2.0))
    dmass   /= (SOLMASS)            # Convert from cgs to solar masses

    return dmass



def apparent_to_absolute(d_pc, mag):
    """
    Converts apparent magnitude to absolute magnitude, given a distance to the object in pc.

    INPUTS
    d_pc: Distance to the object in parsecs.
    mag: Apparent magnitude.

    OUTPUT
    absMag: Absolute magnitude.
    """

    absMag = mag - 5.0 * math.log10(d_pc / 10.0)
    return absMag

#----------------------------------------------DEPENDENT FUNCTIONS-----------------------------------------------
# A function is considered dependent if it utilizes either the above independent functions, or the classes below.
def look(obs, model=None, jobn=None, save=0, savepath=figurepath, colkeys=None, diskcomb=0, msize=7.0, xlim=[2e-1, 2e3], ylim=[1e-15, 1e-9], params=1, leg=1, public=0,odustonly = 0):
    """
    Creates a plot of a model and the observations for a given target.

    INPUTS
    model: The object containing the target's model. Should be an instance of the TTS_Model class. This is an optional input.
    obs: The object containing the target's observations. Should be an instance of the TTS_Obs class.
    jobn: The job number you want to use when you save the plot, if different than the one listed in the model.
    save: BOOLEAN -- If 1 (True), will save the plot in a pdf file. If 0 (False), will output to screen.
    savepath: The path that a saved PDF file will be written to. This is defaulted to the hard-coded figurepath at top of this file.
    colkeys: An optional input array of color strings. This can be used to overwrite the normal color order convention. Options include:
             p == purple, r == red, m == magenta, b == blue, c == cyan, l == lime, t == teal, g == green, y == yellow, o == orange,
             k == black, w == brown, v == violet, d == gold, n == pumpkin, e == grape, j == jeans, s == salmon
             If not specified, the default order will be used, and once you run out, we'll have an error. So if you have more than 18
             data types, you'll need to supply the order you wish to use (and which to repeat). Or you can add new colors using html tags
             to the code, and then update this header.
    diskcomb: BOOLEAN -- If 1 (True), will combine outer wall and disk components into one for plotting. If 0 (False), will separate.
    xlim: A list containing the lower and upper x-axis limits, respectively. Has default values.
    ylim: A list containing the lower and upper y-axis limits, respectively. Has default values.
    params: BOOLEAN -- If 1 (True), the parameters for the model will be printed on the plot.
    leg: BOOLEAN -- If 1 (True), the legend will be printed on the plot.

    OUTPUT
    A plot. Can be saved or plotted to the screen based on the "save" input parameter.
    """

    photkeys            = obs.photometry.keys()         # obs.photometry and obs.spectra are dictionaries.
    speckeys            = obs.spectra.keys()
    colors              = {'p':'#7741C8', 'r':'#F50C0C', 'm':'#F50CA3', 'b':'#2B0CF5', 'c':'#0CE5F5', 'l':'#33F50C', 't':'#4DCE9B', \
                           'g':'#1D5911', 'y':'#BFB91E', 'o':'#F2A52A', 'k':'#060605', 'w':'#5A3A06', 'v':'#BD93D2', 'd':'#FFD900', \
                           'n':'#FF7300', 'e':'#9A00FA', 'j':'#00AAFF', 's':'#D18787'}
    if colkeys == None:
        colkeys         = ['p', 'r', 'o', 'b', 'c', 'm', 'g', 'y', 'l', 'k', 't', 'w', 'v', 'd', 'n', 'e', 'j', 's']    # Order in which colors are used

    # Let the plotting begin!
    if save == 0:
        plt.clf()
    plt.figure(1)

    # Plot the spectra first:
    for sind, skey in enumerate(speckeys):
        if 'err' not in obs.spectra[skey].keys():
            plt.plot(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], 'o', mew=1.0, markersize=3, \
                     mfc=colors[colkeys[sind]], mec= colors[colkeys[sind]], label=skey)
        else:
            plt.errorbar(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], yerr=obs.spectra[skey]['err'], \
                         mec=colors[colkeys[sind]], fmt='o', mfc=colors[colkeys[sind]], mew=1.0, markersize=2, \
                         ecolor=colors[colkeys[sind]], elinewidth=0.5, capsize=1.0, label=skey)

    # Next is the photometry:
    for pind, pkey in enumerate(photkeys):
        # If an upper limit only:
        if pkey in obs.ulim:
            plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'v', \
                     color=colors[colkeys[pind+len(speckeys)]], markersize=msize, label=pkey, zorder=pind+10)
        # If not an upper limit, plot as normal:
        else:
            if 'err' not in obs.photometry[pkey].keys():
                plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'o', mfc='w', mec=colors[colkeys[pind+len(speckeys)]], mew=1.0,\
                         markersize=msize, label=pkey, zorder=pind+10)
            else:
                plt.errorbar(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], yerr=obs.photometry[pkey]['err'], \
                             mec=colors[colkeys[pind+len(speckeys)]], fmt='o', mfc='w', mew=1.0, markersize=msize, \
                             ecolor=colors[colkeys[pind+len(speckeys)]], elinewidth=2.0, capsize=3.0, label=pkey, zorder=pind+10)
    # Publication style?
    if public:
        # Now, the model (if a model supplied):
        if model != None:
            modkeys         = model.data.keys()
            if 'phot' in modkeys:
                plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
            # Will be combining the inner/outer walls with the disk emission component:
            if 'dust' in modkeys:
                if 'owall' in modkeys:
                    if 'newIWall' in model.__dict__:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.newIWall + model.newOWall + model.data['disk'] + model.data['dust']
                        else:
                            diskflux = model.newIWall + model.data['owall'] + model.data['disk'] + model.data['dust']
                    else:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.data['iwall'] + model.newOWall + model.data['disk'] + model.data['dust']
                        else:
                            diskflux = model.data['iwall'] + model.data['owall'] + model.data['disk'] + model.data['dust']
                else:
                    if 'newIWall' in model.__dict__:
                        diskflux = model.newIWall + model.data['disk'] + model.data['dust']
                    else:
                        diskflux = model.data['iwall'] + model.data['disk'] + model.data['dust']
            else:
                if 'owall' in modkeys:
                    if 'newIWall' in model.__dict__:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.newIWall + model.newOWall + model.data['disk']
                        else:
                            diskflux = model.newIWall + model.data['owall'] + model.data['disk']
                    else:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.data['iwall'] + model.newOWall + model.data['disk']
                        else:
                            diskflux = model.data['iwall'] + model.data['owall'] + model.data['disk']
                else:
                    if 'newIWall' in model.__dict__:
                        diskflux = model.newIWall + model.data['disk']
                    else:
                        diskflux = model.data['iwall'] + model.data['disk']
            plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
            if 'scatt' in modkeys:
                plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')
            if 'shock' in modkeys:
                plt.plot(model.data['WTTS']['wl'], model.data['WTTS']['lFl'], c='b', linewidth=2.0, zorder=1, label='WTTS Photosphere')
                plt.plot(model.data['shock']['wl'], model.data['shock']['lFl'], c=colors['j'], linewidth=2.0, zorder=2, label='MagE')
                plt.plot(model.data['shockLong']['wl'], model.data['shockLong']['lFl'], c=colors['s'], linewidth=2.0, zorder=2, label='Shock Model')
            if 'total' in modkeys:
                plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
    else:
        # Now, the model (if a model supplied):
        if model != None:
            modkeys         = model.data.keys()
            if 'phot' in modkeys:
                plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
            if 'dust' in modkeys:
                plt.plot(model.data['wl'], model.data['dust'], ls='--', c='#F80303', linewidth=2.0, label='Opt. Thin Dust')
                
            if odustonly == False:
                if 'idisk' in modkeys:
                    plt.plot(model.data['wl'], model.data['idisk'], ls ='--', c = '#f8522c', linewidth = 2.0, label = 'Inner Disk')
                if 'odisk' in modkeys:
                    plt.plot(model.data['wl'], model.data['odisk'], ls ='--', c = '#024747', linewidth = 2.0, label = 'Outer Disk')
                if 'iwall' in modkeys:
                    if 'owall' in modkeys:
                        label = 'Inner Wall'
                    else:
                        label = 'Wall'
                    try:
                        plt.plot(model.data['wl'], model.newIWall, ls='--', c='#53EB3B', linewidth=2.0, label=label)
                    except AttributeError:
                        plt.plot(model.data['wl'], model.data['iwall'], ls='--', c='#53EB3B', linewidth=2.0, label=label)
                if diskcomb:
                    try:
                        diskflux     = model.newOwall + model.data['disk']
                    except AttributeError:
                        try:
                            diskflux = model.data['owall'] + model.data['disk']
                        except KeyError:
                            print('LOOK: Error, tried to combine outer wall and disk components but one component is missing!')
                        else:
                            plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Outer Disk')
                else:
                    try:
                        plt.plot(model.data['wl'], model.newOWall, ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
                    except AttributeError:
                        if 'owall' in modkeys:
                            plt.plot(model.data['wl'], model.data['owall'], ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
                    if 'disk' in modkeys:
                        plt.plot(model.data['wl'], model.data['disk'], ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
                        
                if 'scatt' in modkeys:
                    plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')
                if 'shock' in modkeys:
                    plt.plot(model.data['WTTS']['wl'], model.data['WTTS']['lFl'], c='b', linewidth=2.0, zorder=1, label='WTTS Photosphere')
                    plt.plot(model.data['shock']['wl'], model.data['shock']['lFl'], c=colors['j'], linewidth=2.0, zorder=2, label='MagE')
                    plt.plot(model.data['shockLong']['wl'], model.data['shockLong']['lFl'], c=colors['s'], linewidth=2.0, zorder=2, label='Shock Model')
            if 'total' in modkeys:
                plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
    # Now, the relevant meta-data:
    if model != None:
        if params:
            if odustonly == False:
                plt.figtext(0.60,0.88,'Eps = '+ str(model.eps), color='#010000', size='9')
                plt.figtext(0.80,0.88,'Alpha = '+ str(model.alpha), color='#010000', size='9')
                plt.figtext(0.60,0.82,'Amax = '+ str(model.amax), color='#010000', size='9')
                plt.figtext(0.60,0.85,'Rin = '+ str(model.rin), color='#010000', size='9')
                plt.figtext(0.80,0.85,'Rout = '+ str(model.rdisk), color='#010000', size='9')
                plt.figtext(0.60,0.79,'Altinh = '+ str(model.wallH), color='#010000', size='9')
                plt.figtext(0.80,0.82,'Mdot = '+ str(model.mdot), color='#010000', size='9')
                # If we have an outer wall height:
                try:
                    plt.figtext(0.80,0.79,'AltinhOuter = '+ str(model.owallH), color='#010000', size='9')
                except AttributeError:
                    plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.temp), color='#010000', size='9')
                else:
                    plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.itemp), color='#010000', size='9')
                    plt.figtext(0.80,0.76,'OWall Temp = '+ str(model.temp), color='#010000', size='9')
                if 'idisk' in modkeys:
                    plt.figtext(0.60, 0.73, 'IDisk Rout = '+str(model.irdisk), color = '#010000', size = '9')
                    plt.figtext(0.80, 0.73, 'IDisk Jobn = '+str(model.ijobn), color = '#010000', size = '9')
             
    # Lastly, the remaining parameters to plotting (mostly aesthetics):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel(r'${\rm \lambda F_{\lambda}\; (erg\; s^{-1}\; cm^{-2})}$')
    plt.xlabel(r'${\rm {\bf \lambda}\; (\mu m)}$')
    plt.title(obs.name.upper())
    if leg:
        plt.legend(loc=3, numpoints = 1)

    # Should we save or should we plot?
    if save:
        if jobn == None:
            try:
                jobstr      = str(model.jobn).zfill(model.fill)
            except AttributeError:
                plt.savefig(savepath + obs.name.upper() + '_obsdata' + '.pdf', dpi=300)
            else:
                plt.savefig(savepath + obs.name.upper() + '_' + jobstr + '.pdf', dpi=300)
        else:
            try:
                jobstr      = str(jobn).zfill(model.fill)
            except AttributeError:
                plt.savefig(savepath + obs.name.upper() + '_obsdata' + '.pdf', dpi=300)
            else:
                plt.savefig(savepath + obs.name.upper() + '_' + jobstr + '.pdf', dpi=300)
        plt.clf()
    else:
        plt.show()

    return

def searchJobs(target, dpath=datapath, **kwargs):
    """
    Searches through the job file outputs to determine which jobs (if any) matches the set of input parameters.

    INPUTS
    target: The name of the target we're checking against (e.g., cvso109, DMTau, etc.).
    **kwargs: Any keyword arguments (kwargs) supplied. These should correspond to the header filenames (not case sensitive). The code
              will loop through each of these kwargs and see if they all match.

    OUTPUTS
    job_matches: A numpy array containing all the jobs that matched the kwargs. Can be an empty array, single value array, or
                 multivalued array. Will contain matches by their integer number.
    """

    print('THIS MIGHT BE DEFUNCT!')

    job_matches         = np.array([], dtype='str')
    targList            = filelist(dpath)

    # Pop out all files that do not correspond to jobs:
    not_data            = []
    for f in targList:
        if f.startswith(target+'_') and f.endswith('.fits'):
            continue
        else:
            not_data.append(targList.index(f))
    for ind, val in enumerate(not_data):
        targList.pop(val - ind)

    # Now go through the list and find any jobs matching the desired input parameters:
    for jobi, job in enumerate(targList):
        if 'OTD' in job:
            continue
        fitsF           = fits.open(dpath+job)
        header          = fitsF[0].header
        for kwarg, value in kwargs.items():
            if header[kwarg.upper()] != value:
                break
        else:
            # Check if three or four string number:
            if job[-9] == '_':
                job_matches = np.append(job_matches, job[-8:-5])
            else:
                job_matches = np.append(job_matches, job[-9:-5])
        fitsF.close()

    return job_matches

def loadPickle(name, picklepath=datapath, num=None, red=0, fill = 3, py2 = False):
    """
    Loads in a pickle saved from the TTS_Obs class.
    
    INPUTS
    name: The name of the object whose observations are stored in the pickle.
    picklepath: The directory location of pickle. Default path is datapath, defined at top of this module.
    num: An optional number provided if there are multiple pickles for this object and you want to load a specific one.
    red: If loading in a red object use this
    fill: Zero padding on job numbers
    py2: If using pickles created with python2 set this flag to True
    
    OUTPUT
    pickle: The object containing the data loaded in from the pickle.
    """
    
    if py2:
        if red:
            if num == None:
                # Check if there is more than one
                flist           = filelist(picklepath)
                if (name + '_red_1.pkl') in flist:
                    print('LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!')
                f               = open(picklepath+name+'_red.pkl', 'rb')
                pickle          = cPickle.load(f, encoding = 'latin1')
                f.close()
            elif num != None:
                f               = open(picklepath+name+'_red_'+str(num).zfill(fill)+'.pkl', 'rb')
                pickle          = cPickle.load(f, encoding = 'latin1')
                f.close()
            return pickle
        else:
            if num == None:
                # Check if there is more than one
                flist           = filelist(picklepath)
                if (name + '_obs_1.pkl') in flist:
                    print('LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!')
                f               = open(picklepath+name+'_obs.pkl', 'rb')
                pickle          = cPickle.load(f, encoding = 'latin1')
                f.close()
            elif num != None:
                f               = open(picklepath+name+'_obs_'+str(num).zfill(fill)+'.pkl', 'rb')
                pickle          = cPickle.load(f, encoding = 'latin1')
                f.close()
        
    else:
        if red:
            if num == None:
                # Check if there is more than one
                flist           = filelist(picklepath)
                if (name + '_red_1.pkl') in flist:
                    print('LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!')
                f               = open(picklepath+name+'_red.pkl', 'rb')
                pickle          = cPickle.load(f)
                f.close()
            elif num != None:
                f               = open(picklepath+name+'_red_'+str(num).zfill(fill)+'.pkl', 'rb')
                pickle          = cPickle.load(f)
                f.close()
            return pickle
        else:
            if num == None:
                # Check if there is more than one
                flist           = filelist(picklepath)
                if (name + '_obs_1.pkl') in flist:
                    print('LOADPICKLE: Warning! There is more than one pickle file for this object! Make sure it is the right one!')
                f               = open(picklepath+name+'_obs.pkl', 'rb')
                pickle          = cPickle.load(f)
                f.close()
            elif num != None:
                f               = open(picklepath+name+'_obs_'+str(num).zfill(fill)+'.pkl', 'rb')
                pickle          = cPickle.load(f)
                f.close()
        
    return pickle

def job_file_create(jobnum, path, fill=3, iwall=0, **kwargs):
    """
    Creates a new job file that is used by the D'Alessio Model.
    
    INPUTS
    jobnum: The job number used to name the output job file.
    path: The path containing the sample job file, and ultimately, the output.
    fill: Pads the output file such that the name will be jobXXX if 3, jobXXXX if 4, etc.
    iwall: BOOLEAN -- if True (1), output will turn off switches so we just run as inner wall.
    **kwargs: The keywords arguments used to make changes to the sample file. Available
              kwargs include:
        amaxs - maximum grain size in disk
        epsilon - settling parameter
        mstar - mass of protostar
        tstar - effective temperature of protostar
        rstar - radius of protostar
        dist - distance to the protostar (or likely, the cluster it's in)
        mdot - the mass accretion rate of protostellar system
        mdotstar - the mass accretion rate onto the star. Usually same as mdot but not required.
        tshock - the temperature of the shock
        alpha - the alpha viscosity parameter
        mui - the cosine of the inclination angle
        rdisk - the outer radius of the disk
        labelend - the labelend of all output files when job file is run
        temp - the temperature of the inner wall
        altinh - the height of the inner wall in scale heights
        fracolive - the fractional abundance of amorphous olivine
        fracpyrox - the fractional abundance of amorphous pyroxene
        fracforst - the fractional abundance of crystalline forsterite
        fracent - the fractional abundance of crystalline enstatite
        lamaxb - string for maximum grain size in the disk midplane (currently accepts '1mm' and '1cm')
        amaxw - maximum grain size in the wall. If not supplied, code will assume that it is the same as the the grain size in the disk
        d2g - Dust to gas mass ratio
        
        Some can still be included, such as dust grain compositions. They just aren't
        currently supported. If any supplied kwargs are unused, it will print at the end.
    
    OUTPUT
    A job file with the name jobXXX, where XXX is the three-string number from 001 - 999. If
    No formal outputs are returned by this function; the file is created in the path directory.
    """
    
    # First we have to make sure that the job_sample file has been "fixed" for the \r issue:
    os.system("cat " + path + "job_sample | tr -d '\r' > " + path + "job_sample2")
    os.system("mv " + path + "job_sample2 " + path + "job_sample")
    
    # Next, let's read in the sample job file so we have a template:
    job_file = open(path+'job_sample', 'r')
    fullText = job_file.readlines()     # All text in a list of strings
    job_file.close()
    
    text = ''.join(fullText)
    
    if len(fullText) == 0:
        raise ValueError('JOB_FILE_CREATE: job_sample file missing/empty!')
    
    #First change alpha is present in kwargs
    if 'amaxs' in kwargs:
        #Break grain size into something parsable
        amaxVal = kwargs['amaxs']
        if amaxVal != 10 and amaxVal != 100:
            amaxStr = str(amaxVal)
        elif amaxVal == 10:
            amaxStr = '10'
        elif amaxVal == 100:
            amaxStr = '100'
        
        #Add a # to the one that was missing one
        start = text.find('\nset AMAXS=')
        text = text[:start+1]+'#'+text[start+1:]
        
        start = text[start:].find('\nset lamaxs') + start
        text = text[:start+1]+'#'+text[start+1:]
        
        #Now remove the # for the selected grain size
        start = text.find("#set AMAXS='"+amaxStr)
        text = text[:start] + text[start+1:]
        
        start = text[start:].find('#set lamaxs=') + start
        text = text[:start] + text[start+1:]
        
        del kwargs['amaxs']
    
    
    #Now handle the case of the wall having different grain sizes.
    if 'amaxw' in kwargs:
        #Break grain size into something parsable
        amaxwVal = kwargs['amaxw']
        
        
        if amaxwVal != 10 and amaxwVal != 100:
            amaxwStr = str(amaxwVal)
        elif int(amaxwVal) == 10:
            amaxwStr = '10'
        elif amaxwVal == 100:
            amaxwStr = '100'
        
        #Add a # to the one that was missing one
        start = text.find('\nset AMAXW=')
        text = text[:start+1]+'#'+text[start+1:]
        
        start = text[start:].find('\nset lamaxw') + start
        text = text[:start+1]+'#'+text[start+1:]
        
        #Now remove the # for the selected grain size
        start = text.find("#set AMAXW='"+amaxwStr)
        text = text[:start] + text[start+1:]
        
        start = text[start:].find('#set lamaxw=') + start
        text = text[:start] + text[start+1:]
        
        del kwargs['amaxw']
        
    
    #Handles the case where amaxw is not supplied by assuming making it the same as amax
    else:
        #Add a # to the one that was missing one
        start = text.find('\nset AMAXW=')
        text = text[:start+1]+'#'+text[start+1:]
        
        start = text[start:].find('\nset lamaxw') + start
        text = text[:start+1]+'#'+text[start+1:]
        
        #Now remove the # for the selected grain size
        start = text.find("#set AMAXW=$AMAXS")
        text = text[:start] + text[start+1:]
        
        start = text[start:].find('#set lamaxw=') + start
        text = text[:start] + text[start+1:]
        
        
    
    # Now, we examine the epsilon parameter if a value provided:
    if 'epsilon' in kwargs:
        epsVal = kwargs['epsilon']
        epsStr = str(epsVal)
        
        #Add # to the one that was missing one
        start = text.find('\nset EPS=')
        text = text[:start+1]+'#'+text[start+1:]
        
        start = text[start:].find('\nset epsilonbig') + start
        text = text[:start+1]+'#'+text[start+1:]
        
        #Now remove the # for the selected grain size
        start = text.find("#set EPS='"+epsStr)
        text = text[:start] + text[start+1:]
        
        start = text[start:].find('#set epsilonbig=') + start
        text = text[:start] + text[start+1:]
        
        del kwargs['epsilon']
        
    
    #Now go through the rest of the parameters
    dummykwargs = copy.deepcopy(kwargs)
    for param in dummykwargs:
        
        #Remove the used kwarg
        del kwargs[param]
        
        #Set up the parameter
        paramstr = str(dummykwargs[param])
        if param != 'labelend' and param != 'lamaxb':
            param  = param.upper()
        if param == 'DIST':
            param = 'DISTANCIA'
            
        #Fix the special case of lamaxb
        if param == 'lamaxb':
            amaxdict={'1mm':'1000','1cm':'10000'}
            paramstr = amaxdict[dummykwargs[param]]
            
            start = text.find('set '+param+"='") + len('set '+param+"='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start]+'amax'+dummykwargs[param]+text[end:]
            
            start = text.find("set AMAXB='") + len("set AMAXB='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start]+paramstr +text[end:]
        
        #Fix the special case of temp + Tshock
        elif param == 'TEMP' or param == 'TSHOCK':
            start = text.find('set '+param+"=") + len('set '+param+"=")
            end = start + len(text[start:].split(".")[0])
            text = text[:start]+paramstr+text[end:]
        #Fix the special case of altinh
        elif param == 'ALTINH':
            start = text.find('set '+param+'=') + len('set '+param+'=')
            end = start + len(text[start:].split("#")[0])
            text = text[:start]+paramstr+'    '+text[end:]
        #Fix the special case of MDOTSTAR (Sometimes it is $MDOT)
        elif param == 'MDOTSTAR':
            start = text.find('set '+param+'=') + len('set '+param+'=')
            end = start + len(text[start:].split("#")[0])
            text = text[:start]+"'"+paramstr+"'"+' '+text[end:]
        elif param == 'D2G':
            start = text.find('set '+param+'=') + len('set '+param+'=')
            end = start + len(text[start:].split("\n")[0])
            text = text[:start]+paramstr+text[end:]
        
        #Change the rest
        else:
            #Fix some names
            if param == 'FRACOLIVE':
                param = 'AMORPFRAC_OLIVINE'
            elif param == 'FRACPYROX':
                param = 'AMORPFRAC_PYROXENE'
            elif param == 'FRACFORST':
                param = 'FORSTERITE_FRAC'
            elif param == 'FRACENT':
                param = 'ENSTATITE_FRAC'
            
            start = text.find('set '+param+"='") + len('set '+param+"='")
            end = start + len(text[start:].split("'")[0])
            
            #Replace the parameter
            text = text[:start]+paramstr+text[end:]
    
    if iwall:
        
        turnoff = ['IPHOT', 'IOPA', 'IVIS', 'IIRR', 'IPROP', 'ISEDT']
        
        for switch in turnoff:
            start = text.find('set '+switch+"='") + len('set '+switch+"='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + '0' + text[end:]
    
    outtext = [s + '\n' for s in text.split('\n')]
    
    # Once all changes have been made, we just create a new job file:
    string_num  = str(jobnum).zfill(fill)
    newJob      = open(path+'job'+string_num, 'w')
    newJob.writelines(outtext)
    newJob.close()
    
    # Lastly, check for unused kwargs that may have been misspelled:
    if len(kwargs) != 0:
        print('JOB_FILE_CREATE: Unused kwargs, could be mistakes:')
        print(kwargs.keys())
    
    return
    
def job_optthin_create(jobn, path, fill=3, **kwargs):
    """
    Creates a new optically thin dust job file.

    INPUTS
    jobn: The job number used to name the output job file.
    path: The path containing the sample job file, and ultimately, the output.
    fill: Pads the output file such that the name will be job_optthinXXX if 3, job_optthinXXXX if 4, etc
    **kwargs: The keywords arguments used to make changes to the sample file. Available
              kwargs include:
        amax - maximum grain size
        tstar - effective temperature of protostar
        rstar - radius of protostar
        dist - distance to the protostar (or likely, the cluster it's in)
        mui - the cosine of the inclination angle
        rout - the outer radius
        rin - the inner radius
        labelend - the labelend of all output files when job file is run
        tau - optical depth, I think
        power - no idea what this one is
        fudgeorg - don't know this one either
        fudgetroi - or this one...should probably look this up
        fracsil - fraction of silicates by mass
        fracent - fraction of enstatite by mass
        fracforst - fraction of forsterite by mass
        fracamc - fraction of amorphous carbon by mass
        fracice - fraction of ice by mass (I assume?)

        Some can still be included, such as dust grain compositions. They just aren't
        currently supported. If any supplied kwargs are unused, it will print at the end.

    OUTPUT
    A job file with the name job_optthinXXX, where XXX is the three-string number from 001 - 999. If
    No formal outputs are returned by this function; the file is created in the path directory.
    """

    # First, load in the sample job file for a template:
    job_file = open(path+'job_optthin_sample', 'r')
    fullText = job_file.readlines()     # All text in a list of strings
    job_file.close()

    text = ''.join(fullText)

    # Now we run through the list of changes desired and change them:
    # If we want to change amax:
    if 'amax' in kwargs:
        #Break the grain size into something parsable
        amaxVal = kwargs['amax']
        del kwargs['amax']
        if amaxVal != 10 and amaxVal != 100 and amaxVal != 1000:
            amaxStr = str(amaxVal)
            start = amaxStr.find('.')
            amaxStr = amaxStr[:start]+'p'+amaxStr[start+1:]
        elif amaxVal == 10:
            amaxStr = '10'
        elif amaxVal == 100:
            amaxStr = '100'
        elif amaxVal == 1000:
            amaxStr = '1000'

        #add a pound sign to the one that was missing one.
        start = text.find('\nset lamax')
        text = text[:start+1]+'#'+text[start+1:]

        #Remove the pound sign for the grain size we want
        start = text.find("#set lamax='amax"+amaxStr+"'")
        text = text[:start] + text[start+1:]

    # Now we can cycle through the easier changes desired:
    dummykwargs = copy.deepcopy(kwargs)
    for param in dummykwargs:

        #Remove the used kwarg
        del kwargs[param]

        #Set up the parameter
        paramstr = str(dummykwargs[param])

        if param != 'labelend':
            param  = param.upper()
        if param == 'DIST':
            param = 'DISTANCIA'
        if param == 'TAU':
            param = 'TAUMIN'

        #Find region of text to replace
        start = text.find('set '+param+"='") + len('set '+param+"='")
        end = start + len(text[start:].split("'")[0])

        #Replace the parameter
        text = text[:start]+paramstr+text[end:]

    #Turn the text into something that can be written out:
    outtext = [s + '\n' for s in text.split('\n')]

    # Once all changes have been made, we just create a new optthin job file:
    string_num  = str(jobn).zfill(fill)
    newJob      = open(path+'job_optthin'+string_num, 'w')
    newJob.writelines(outtext)
    newJob.close()

    # Lastly, check for unused kwargs that may have been misspelled:
    if len(kwargs) != 0:
        print('JOB_OPTTHIN_CREATE: Unused kwargs, could be mistakes:')
        print(kwargs.keys())

    return

def create_runall(jobstart, jobend, clusterpath, optthin = False, outpath = '', commonpath = commonpath, fill = 3):
    '''
    create_runall()
    
    INPUTS:
        jobstart: [int] First job file in grid
        jobsend: [int] Last job file in grid
    
    OPTIONAL INPUTS:
        optthin: [Boolean] Set to True for optically thin dust models.
        outpath: [String] Location of where the runall script should be sent. Default is current directory.
        edgepath: [String] Path to where the runall_template file is located. Default is the edge director
    '''
    #Now write the runall script
    runallfile = open(commonpath+'runall_template', 'r')
    fulltext = runallfile.readlines()     # All text in a list of strings
    runallfile.close()
    
    #Turn it into one large string
    text = ''.join(fulltext)
    
    #Replace the path
    start = text.find('cd ')+len('cd ')
    end = start +len(text[start:].split('\n')[0])
    text = text[:start] + clusterpath + text[end:]
    
    #Replace the jobstart
    start = text.find('#qsub -t ')+len('#qsub -t ')
    end = start +len(text[start:].split('-')[0])
    text = text[:start] + str(int(jobstart)) + text[end:]
    
    #Replace the job end
    start = text.find('#qsub -t '+str(int(jobstart))+'-')+len('#qsub -t '+str(int(jobstart))+'-')
    end = start +len(text[start:].split(' runall.csh')[0])
    text = text[:start] + str(int(jobend)) + text[end:]
    
    #Replace fill
    start = text.find('job%0')+len('job%0')
    end = start +len(text[start:].split('d" $SGE_TASK_ID')[0])
    text = text[:start] + str(int(fill)) + text[end:]
    
    #If the job is optically thin, replace job
    if optthin:
        start = text.find('job%0')
        end = start+len('job%0')
        text = text[:start]+'job_optthin%0'+text[end:]
    
    #Turn the text back into something that can be written out
    outtext = [s + '\n' for s in text.split('\n')]
    
    #Write out the runall file
    newrunall = open(outpath+'runall.csh', 'w')
    newrunall.writelines(outtext)
    newrunall.close()
    


def model_rchi2(obj, model, obsNeglect=[], wp=0.0, non_reduce=1, verbose = 1):
    """
    Calculates a reduced chi-squared goodness of fit.

    INPUTS
    obj: The object containing the observations we are comparing to. Is an instance of TTS_Obs()
    model: The model to test. Must be an instance of TTS_Model(), with a calculated total.
    obsNeglect: A list of all the observations keys that you don't wish to be considered.
    wp: The weight option you'd like to use for your photometry's chi2 calculation. The
            weight for the spectra will just be 1 - wp. Default is .5 for each.
    non_reduce: BOOLEAN -- if True (1), will calculate a normal chi squared (not reduced).
                If this is the case, the weighting will be 1 for the photometry, and for the
                spectra will be the number of photometric points / number of spectra points.

    OUTPUT
    total_chi: The value for the reduced chi-squared test on the model.

    NOTE:
    If synthetic fluxes want to be used when computing the chi square, these keywords
    need to be used when adding photometry:
    'CVSO', 'IRAC', 'MIPS', 'CANA', 'PACS', 'WISE', '2MASS'

    New instruments can be added with time. If any instrument is added,
    the instrKeylist should be updated in calc_filters and in add_photometry.
    """

    #First check to see if there is a total component
    if 'total' not in model.data:
        if verbose:
            print("MODEL_RCHI2: Model "+model.jobn+" does not have 'total' values, returning...")
        return -1

    # We compute the chi2 for the photometry and the spectra separately.
    # Start with photometry:
    chiSF = []
    chiP = []
    wavelength = []
    flux = []
    errs = []
    max_lambda = 0.0
    min_lambda = 1e10 # A big number
    
    #If difference in wavelengths of instruments in photometry and calc_filters are greater than this, code will raise an error
    threshold = 0.1 
    
    for obsKey in obj.photometry.keys():
        if obsKey in obsNeglect:
            # In order to be safe, we reset phot_dens to 0 when we discard some photometry
            obj.phot_dens = 0.0
            continue                            # Skip any data you want to neglect
        if obsKey in obj.ulim:
            continue                            # Skip any upper limits
        if obj.phot_dens == 0.0: # We do this just if phot_dens has not been computed and saved yet
            # Maximum lambda, used later
            if max(obj.photometry[obsKey]['wl']) > max_lambda:
                max_lambda = max(obj.photometry[obsKey]['wl'])
            # Minimum lambda, used later
            if min(obj.photometry[obsKey]['wl']) < min_lambda:
                min_lambda = min(obj.photometry[obsKey]['wl'])
        # For the instruments with synthetic fluxes
        if obsKey in model.synthFlux.keys():
            for ind,wl in enumerate(obj.photometry[obsKey]['wl']):
                l = np.argmin(abs(model.synthFlux[obsKey]['wl']-wl)) # We find the band
                
                #Raise an error if the wavelengths are larger than the threshold given above.
                if abs((model.synthFlux[obsKey]['wl'][l] - wl)/wl) > threshold:
                    raise ValueError('MODEL_RCHI2: Error with wavelength for '+obsKey+'. Check wavelength for photometry')
                obs_flux = obj.photometry[obsKey]['lFl'][ind]
                model_flux = model.synthFlux[obsKey]['lFl'][l]
                try:
                    obs_err = obj.photometry[obsKey]['err'][ind]
                    chiSF.append((obs_flux - model_flux)/ obs_err)
                except:
                    # If there is no value for the error or it is 0, assume 10%
                    chiSF.append((obs_flux - model_flux)/ (0.1 * obs_flux))
        else:
        # If we don't have a synthetic flux for that instrument, we will
        # have to do some more work
            wavelength.append(obj.photometry[obsKey]['wl'])
            flux.append(obj.photometry[obsKey]['lFl'])
            try:
                # Fix if error is NaN
                if np.isnan(np.sum(obj.photometry[obsKey]['err'])):
                    badErr = np.where(np.isnan(obj.photometry[obsKey]['err']))
                    newErrs = obj.photometry[obsKey]['err']
                    newErrs[badErr] = 0.1*obj.photometry[obsKey]['lFl'][badErr]
                    errs.append(newErrs)
                else:
                    errs.append(obj.photometry[obsKey]['err'])
            except KeyError:
                # if no error, assume 10%:
                errs.append(0.1 * obj.photometry[obsKey]['lFl'])

    # Calculate the chi2 for the instruments with synthetic fluxes
    chiSF = np.array(chiSF)
    rchi_sqSF   = np.sum(chiSF*chiSF)

    # For the instruments that do not have synthetic fluxes
    if len(wavelength) > 0:
        wavelength = np.concatenate(wavelength)
        flux = np.concatenate(flux)
        errs = np.concatenate(errs)
    
        # Check and remove NaNs from the data, if any:
        if np.isnan(np.sum(flux)):
            badVals    = np.where(np.isnan(flux))   # Where the NaNs are located
            flux       = np.delete(flux, badVals)
            wavelength = np.delete(wavelength, badVals)
            errs       = np.delete(errs, badVals)
        # If there are NaNs in the actual model, remove them:
        if np.isnan(np.sum(model.data['total'])):
            badValsMod = np.where(np.isnan(model.data['total']))
            for key in model.data.keys():
                model.data[key] = np.delete(model.data[key], badValsMod)
    
        # Sort the arrays:
        waveindex      = np.argsort(wavelength)     # Indices that sort the array
        wavelength     = wavelength[waveindex]
        flux           = flux[waveindex]
        errs           = errs[waveindex]
    
        # Interpolate the model so the observations and model are on the same grid:
        modelFlux      = np.interp(wavelength, model.data['wl'], model.data['total'])
    
        # Calculate the chi for the instruments without synthetic fluxes
        chiP = (flux - modelFlux) / errs
        # The total chi2 for the photometry will be
        rchi_sqP   = np.sum(chiP*chiP) + rchi_sqSF
    
    #If no instruments with synthetic fluxes then chi2 is just chi2 of the synthetic fluxes
    else:
        rchi_sqP   = rchi_sqSF
    
    # "Density" of photometric points, used later for the spectra weighting calculation
    if obj.phot_dens == 0.0:
    # We save it as an attribute so that it doesn't need to be calculated again
        obj.phot_dens = (len(chiP) + len(chiSF)) / (np.log10(max_lambda) - np.log10(min_lambda))

    # Now, do the same thing but for the spectra:
    if len(obj.spectra.keys()) != 0: # If there is any spectrum
        # Initialize empty lists
        wavelengthS= []
        fluxS      = []
        errsS      = []

        # Build the flux and wavelength vectors, excluding data we don't care about:
        for specKey in obj.spectra.keys():
            if specKey in obsNeglect:
                continue                            # Skip any data you want to neglect
            if wp == 0.0: # If weights are not provided
                if specKey not in obj.spec_dens.keys():
                    # We do this just once, then spec_dens will be saved
                    # Computation of the weights for the spectrum
                    max_lambdaS = max(obj.spectra[specKey]['wl'])
                    min_lambdaS = min(obj.spectra[specKey]['wl'])
                    obj.spec_dens[specKey] = len(obj.spectra[specKey]['wl']) / (np.log10(max_lambdaS) - np.log10(min_lambdaS))
                ws = np.sqrt(obj.phot_dens / obj.spec_dens[specKey])
            else: # If weights are provided, we will use them
                ws = 1.0
            # We save the wavelengths and fluxes in a list (to be converted to an array)
            wavelengthS.append(obj.spectra[specKey]['wl'])
            fluxS.append(obj.spectra[specKey]['lFl'])
            # If we are computing the weights of the spectra, we will use them in the errors            max_lambdaS = max(obj.spectra[specKey]['wl'])
            try:
                # Fix if error is NaN
                if np.isnan(np.sum(obj.spectra[obsKey]['err'])):
                    newErrs = obj.spectra[obsKey]['err']
                    newErrs[np.isnan(obj.spectra[obsKey]['err'])] = 0.1
                    errsS.append(ws * newErrs)
                else:
                    errsS.append(ws * obj.spectra[obsKey]['err'] / obj.spectra[obsKey]['lFl'])
            except KeyError:
                # if no error, assume 10%:
                errsS.append(ws * np.ones(len(obj.spectra[specKey]['wl']))/10.0)

        # Make the lists arrays
        wavelengthS = np.concatenate(wavelengthS)
        fluxS = np.concatenate(fluxS)
        errsS = np.concatenate(errsS)

        # Sort the arrays:
        waveindexS     = np.argsort(wavelengthS)    # Indices that sort the array
        wavelengthS    = wavelengthS[waveindexS]
        fluxS          = fluxS[waveindexS]
        errsS          = errsS[waveindexS]

        # Check and remove NaNs from the data, if any:
        if np.isnan(np.sum(fluxS)):
            badValsS   = np.where(np.isnan(fluxS))  # Where the NaNs are located
            fluxS      = np.delete(fluxS, badValsS)
            wavelengthS= np.delete(wavelengthS, badValsS)
            errsS      = np.delete(errsS, badValsS)

        # Interpolate the model so the observations and model are on the same grid:
        modelFluxS     = np.interp(wavelengthS, model.data['wl'], model.data['total'])

        # chi2 of spectra:
        chiS = (fluxS - modelFluxS) / (errsS*fluxS)
        rchi_sqS = np.sum(chiS*chiS)

        # If weight of spectra was provided, use it now
        if wp != 0.0:
            ws = 1.0 - wp
            rchi_sqS = rchi_sqS * np.sqrt(ws)
            rchi_sqP = rchi_sqP * np.sqrt(wp)

        # Total chi squared
        total_chi = rchi_sqP + rchi_sqS
    else:
        # If there is no spectra, total chi squared will be the photometry chi_sq
        total_chi = rchi_sqP

    # Finally, if we want to reduce the chi_sq, we do it now.
    # (even though it might not make a lot of sense if there are spectra)
    if non_reduce == 0:
        if len(obj.spectra.keys()) != 0: # If there is any spectrum
            total_chi = total_chi / (len(chiP) + len(chiSF) + len(chiS) - 1)
        else:
            total_chi = total_chi / (len(chiP) + len(chiSF) - 1)

    return total_chi                            # Done!


def BIC_Calc(obs, minChi, degFree=6, weight=None, ignoreKeys=[]):
    """
    Calculates the Bayesian Information Criteria (BIC) for your given model.

    INPUTS
    obs: The observations object you're using for the chi-squared calculation.
    minChi: The minimum Chi-Squared value obtained from your grid.
    degFree: The degrees of freedom, i.e., how many model parameters you varied.
    weight: How to calculate the number of points. Currently supports 'TwicePhot', 'SpectraOnly',
            and None (the default). 'TwicePhot' counts twice the number of photometric points.
            'SpectraOnly' counts only the spectral points. The default counts the photometric and
            spectral points together with no weighting given to either.
    ignoreKeys: A list containing any keys you want ignored in the point count.

    OUTPUT
    bic: The Bayesian Information Criteria (BIC) calculated given the inputs.
    """
    # Need to calculate the number of data points being used:
    if weight == 'TwicePhot':
        # If TwicePhot, we count twice the number of photometric points for N
        pointCounter    = 0
        for key in obs.photometry.keys():
            if key in ignoreKeys or key in obs.ulim:
                continue
            try:
                pointCounter    += len(obs.photometry[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
        pointCounter *= 2       # Twice, since the spectra are weighted evenly with photometry
    elif weight == 'ThricePhot':
        # If ThricePhot, we count three times the number of photometric points for N
        pointCounter    = 0
        for key in obs.photometry.keys():
            if key in ignoreKeys or key in obs.ulim:
                continue
            try:
                pointCounter    += len(obs.photometry[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
        pointCounter *= 3       # Three times, once for IRS, once for the rest
    elif weight == 'CVSO':
        # If CVSO, we count the number of photometric points for N and add 25
        pointCounter    = 0
        for key in obs.photometry.keys():
            if key in ignoreKeys or key in obs.ulim:
                continue
            try:
                pointCounter    += len(obs.photometry[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
        pointCounter += 25       # spectra are weighted to be 25
    elif weight == 'SpectraOnly':
        # If SpectraOnly, we count the number of spectral points for N
        pointCounter    = 0
        for key in obs.spectra.keys():
            if key in ignoreKeys:
                continue
            try:
                pointCounter    += len(obs.spectra[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
    elif weight == None:
        # If no weighting (default), we count all of the points, spectra and photometry
        pointCounter    = 0
        for key in obs.photometry.keys():
            if key in ignoreKeys or key in obs.ulim:
                continue
            try:
                pointCounter    += len(obs.photometry[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
        for key in obs.spectra.keys():
            if key in ignoreKeys:
                continue
            try:
                pointCounter    += len(obs.spectra[key]['lFl'])
            except TypeError:   # This error occurs if only one data point present:
                pointCounter    += 1
    else:
        raise IOError('BIC_CALC: You gave an invalid weighting!')

    # Now that we have the number of points (N), we can calculate the BIC:
    bic = minChi + degFree * np.log(pointCounter)

    return bic

def star_param(sptype, mag, Av, dist, params, picklepath=commonpath, jnotv=0):
    """
    Calculates the effective temperature and luminosity of a T-Tauri star. Uses either values based on
    Kenyon and Hartmann (1995), or Pecault and Mamajek (2013). This function is based on code written
    by Alice Perez at CIDA.

    INPUTS
    sptype: The spectral type of your object. Can be either a float value, or an alphanumeric representation.
    mag: The magnitude used for correction. Must be either V band or J band.
    Av: The extinction in the V band.
    dist: The distance to your object in parsecs.
    params: Must be either 'KH' (for Kenyon & Hartmann) or 'PM' (for Pecault and Mamajek)
    picklepath: Where the star_param.pkl file is located. Default is hardcoded for where EDGE.py is located.
    jnotv: BOOLEAN -- if True (1), it sets 'mag' input to be J band magnitude rather than V band.

    OUTPUTS
    Teff: The calculated effective temperature of the star (in Kelvin).
    lum: The calculated luminosity of the star in solar luminosities (L / Lsun).
    """

    # First, we need to load in the pickle containing the conversions:
    stparam_pick = open(picklepath + 'star_param.pkl', 'rb')
    stparam_dict = cPickle.load(stparam_pick)
    stparam_pick.close()

    # Next, create relevant interpolation grids based on desired params:
    # If the spectral type is not a number, we'll need to convert!
    if type(sptype) == float or type(sptype) == int:
        pass
    else:
        sptype = convertSptype(sptype)

    if params == 'KH':
        print('STAR_PARAM: Will be using Kenyon & Hartmann values.')
        tempSpline = sinterp.UnivariateSpline(stparam_dict['KH']['SpType'], stparam_dict['KH']['Teff'], s=0)
        boloSpline = sinterp.UnivariateSpline(stparam_dict['KH']['SpType'], stparam_dict['KH']['BC'], s=0)
    elif params == 'PM':
        print('STAR_PARAM: Will be using Pecaut and Mamajet values.')
        tempSpline = sinterp.UnivariateSpline(stparam_dict['PM']['SpType'], stparam_dict['PM']['Teff'], s=0)
        boloSpline = sinterp.UnivariateSpline(stparam_dict['PM']['SpType'], stparam_dict['PM']['BC'], s=0)
    else:
        raise IOError('STAR_PARAM: Did not enter a valid input for params!')

    # Calculate the effective temperature:
    Teff  = tempSpline(sptype)
    # Error calculation? Do we need to use log base 10?

    # Calculate the luminosity utilizing bolometric correction and distance modulus:
    BCorr = boloSpline(sptype)

    # Check if we have a J mag instead of a V mag:
    if jnotv:
        Mj   = mag + 5 - (5*np.log10(dist)) - 0.29*Av       # Aj/Av = 0.29 (Cardelli, Clayton and Mathis 1989)
        Mbol = Mj + BCorr
    else:
        Mv   = mag + 5 - (5*np.log10(dist)) - Av
        Mbol = Mv + BCorr
    lum = 10.0 ** ((-Mbol+4.74) / 2.5)

    return float(Teff), lum

def normalize(dataDict, normWL, normlFl):
    """
    Normalizes a given spectrum of data to the provided normalization wavelength and flux values. Optionally
    normalizes an associated error array.

    INPUTS
    dataDict: The dictionary containing the data to normalize. Will have 'wl' and 'lFl' keys. 'err' is optional.
    normWL: The wavelength (in same unit as data's wl, typically microns) at which to normalize.
    normlFl: The flux value we are normalizing to at the given normalization wavelength.

    OUTPUT
    normFlux: The normalized flux array.
    normErr: (optional) If errors are included, then this is the normalized errors.
    """

    # Find out if/where the normalization wavelength and flux exist in the data:
    normInd = np.where(dataDict['wl'] >= normWL)[0][0]

    # If the normalization wavelength is between two indices, interpolate the flux:
    if dataDict['wl'][normInd] != normWL:
        # Make sure no NaNs:
        if np.isnan(dataDict['lFl'][normInd]) or np.isnan(dataDict['lFl'][normInd-1]):
            raise ValueError('NORMALIZE: The flux is NaN at the normalization wavelength!')
        if 'err' in dataDict.keys():
            normVal, normErr = (linearInterp(normWL, dataDict['wl'][normInd-1], dataDict['wl'][normInd],
                                             dataDict['lFl'][normInd-1], dataDict['lFl'][normInd],
                                             dataDict['err'][normInd-1], dataDict['err'][normInd]))
        else:
            normVal, normErr = (linearInterp(normWL, dataDict['wl'][normInd-1], dataDict['wl'][normInd],
                                             dataDict['lFl'][normInd-1], dataDict['lFl'][normInd], 0.0, 0.0))
    else:
        if np.isnan(dataDict['lFl'][normInd]):
            raise ValueError('NORMALIZE: The flux is NaN at the normalization wavelength!')
        normVal = dataDict['lFl'][normInd]

    # Now we normalize the flux:
    normFlux = (dataDict['lFl'] / normVal) * normlFl

    # Normalize the error based on percent error:
    if 'err' in dataDict.keys():
        normErr  = (dataDict['err']/dataDict['lFl']) * normFlux
        return normFlux, normErr

    return normFlux

def MdotCalc(Umag, Rmag, d_pc, Temp, Mstar, Rstar):
    """
    Calculates the accretion rate based on the relation in Gullbring et al. 1998, using the
    apparent U band magnitude and some stellar/disk properties.

    INPUTS
    Umag: The U band apparent magnitude for your object.
    Rmag: The R band apparent magnitude for your object.
    d_pc: The distance to the object in parsecs.
    Temp: The stellar effective temperature for your object.
    Mstar: The stellar mass of your object, in units of solar masses.
    Rstar: The stellar radius of your object, in units of solar radii.
    Rin: The inner radius of your disk, in AU.

    OUTPUTS
    Mdot: The mass accretion rate for your object in solar masses per year.
    """

    # Define the arrays containing the temperature, U-R pairs:
    temps   = (np.flipud(np.array([30000, 25400, 22000, 18700, 17000, 15400, 14000, 13000, 11900,
                         10500,  9520,  9230,  8970,  8720,  8460,  8200,  8350,  7850,
                         7580,  7390,  7200,  7050,  6890,  6740,  6590,  6440,  6360,
                         6280,  6200,  6115,  6030,  5945,  5860,  5830,  5800,  5770,
                         5700,  5630,  5520,  5410,  5250,  5080,  4900,  4730,  4590,
                         4350,  4205,  4060,  3850,  3720,  3580,  3470,  3370,  3240,  3050])))
    colors  = (np.flipud(np.array([-1.51, -1.34, -1.18, -1.02, -0.91, -0.78, -0.67, -0.54, -0.42,
                         -0.17,  0.02,  0.1 ,  0.18,  0.26,  0.33,  0.4 ,  0.43,  0.47,
                         0.53,  0.59,  0.64,  0.68,  0.72,  0.75,  0.77,  0.8 ,  0.86,
                         0.94,  1.01,  1.07,  1.14,  1.16,  1.17,  1.24,  1.33,  1.4 ,
                         1.45,  1.52,  1.59,  1.74,  1.91,  2.13,  2.26,  2.63,  2.93,
                         3.21,  3.5 ,  3.79,  3.94,  4.14,  4.19,  4.25,  4.59,  4.87,  5.26])))

    # First, calculate the U band magnitude of the photosphere:
    tempMatch     = np.where(temps == Temp)[0]
    if len(tempMatch) == 0:                         # Is there an exact match? If not, interpolate
        colInterp = np.interp(Temp, temps, colors)
    else:
        colInterp = colors[tempMatch]
    Uphot   = Rmag + colInterp
    print(colInterp)
    # Convert to Absolute Magnitude:
    #uAbsMag = apparent_to_absolute(d_pc, Umag)
    #uPhotAbs= apparent_to_absolute(d_pc, Uphot)

    # Calculate the U flux for the photosphere and star to get excess luminosity:
    photFlux= convertMag(Uphot, 'U') * 1e-3 * (0.068 / 0.367)
    starFlux= convertMag(Umag, 'U') * 1e-3 * (0.068 / 0.367)
    #starFlux= 4.2708040e-14
    #photFlux= convertMag(uPhotAbs, 'U')
    #starFlux= convertMag(uAbsMag, 'U')

    L_u     = (4.0*math.pi) * (starFlux - photFlux) * (d_pc * 3.086e16)**2.0
    L_u_norm= L_u / 3.84e26
    #
    # uAbsMag = apparent_to_absolute(d_pc, uApMag)
    #
    # # Convert the absolute magnitude to a U band luminosity:
    # UmagSun = 5.61      # From Binney and Merrifield 1998
    # #UmagSun = 4.74
    # L_u     = 100.0**((UmagSun - uAbsMag)/5.0)      # Luminosity in U band / Lsun

    # Use the U band luminosity to calculate the accretion luminosity:
    L_acc   = 3.84e26 * 10.0**(1.09*math.log10(L_u_norm) + 0.98)

    # Lastly, back out the accretion rate:
    G       = 6.67e-11                              # G in meters
    #Rin_m   = Rin * 1.496e11                        # Rin in meters
    Rstar_m = Rstar * 6.955e8                       # Rstar in meters
    Mstar_kg= Mstar * 1.989e30                      # Mstar in kg
    Mdot    = (Rstar_m * L_acc / (G*Mstar_kg)) / 0.8 * 3.16e7 / 1.989e30

    return Mdot

def binSpectra(obs, speckeys=[], ppbin=2):
    """
    Bin all the spectra in the obs object corresponding to the supplied
    keys.

    INPUTS
    obs: The observations pickle, which should be an instance if TTS_Obs or Red_Obs.
    speckeys: The keys corresponding to the spectra that should be binned.
    ppbin: The number of points in a given bin.

    OUTPUT
    Though no output is explicitly given, the binned spectra are saved to the observations pickle.
    """

    if len(speckeys) == 0:
        print('BINSPECTRA: No keys were supplied. No binning will occur.')
    else:
        for key in speckeys:
            if key not in obs.spectra.keys():
                print('BINSPECTRA: ' + str(key) + ' not found in the observations object. Skipping.')
                continue
            binnedWL    = np.array([], dtype=float)
            binnedFlux  = binnedWL.copy()
            if len(obs.spectra[key]) == 3:
                binnedErr   = binnedWL.copy()
            for i in range(len(obs.spectra[key]['wl'])):
                if i % ppbin != 0:
                    continue
                if np.isnan(np.sum(obs.spectra[key]['lFl'][i:i+ppbin])):
                    continue
                avgwl   = np.average(obs.spectra[key]['wl'][i:i+ppbin])
                avgflux = np.average(obs.spectra[key]['lFl'][i:i+ppbin])
                binnedWL    = np.append(binnedWL, avgwl)
                binnedFlux  = np.append(binnedFlux, avgflux)
                if len(obs.spectra[key]) == 3:
                    avgerr  = np.std(obs.spectra[key]['lFl'][i:i+ppbin])
                    binnedErr       = np.append(binnedErr, avgerr)
            obs.spectra[key]['wl']  = binnedWL
            obs.spectra[key]['lFl'] = binnedFlux
            if len(obs.spectra[key]) == 3:
                obs.spectra[key]['err'] = binnedErr

    return

#---------------------------------------------------CLASSES------------------------------------------------------
class TTS_Model(object):
    """
    Contains all the data and meta-data for a TTS Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.

    ATTRIBUTES
    name: Name of the object (e.g., CVSO109, V410Xray-2, ZZ_Tau, etc.).
    jobn: The job number corresponding to this model.
    mstar: Star's mass.
    tstar: Star's effective temperature, based on Kenyon and Hartmann 1995.
    rstar: Star's radius.
    dist: Distance to the star.
    mdot: Mass accretion rate.
    mdotstar: Mass accretion rate onto the star. Usually same as mdot but not necessarily.
    alpha: Alpha parameter (from the viscous alpha disk model).
    mui: Inclination of the system.
    rdisk: The outer radius of the disk.
    amax: The "maximum" grain size in the disk. (or just suspended in the photosphere of the disk?)
    eps: The epsilon parameter, i.e., the amount of dust settling in the disk.
    tshock: The temperature of the shock at the stellar photosphere.
    temp: The temperature at the inner wall (1400 K maximum).
    altinh: Scale heights of extent of the inner wall.
    wlcut_an:
    wlcut_sc:
    nsilcomp: Number of silicate compounds.
    siltotab: Total silicate abundance.
    amorf_ol:
    amorf_py:
    forsteri: Forsterite Fractional abundance.
    enstatit: Enstatite Fractional abundance.
    rin: The inner radius in AU.
    dpath: Path where the data files are located.
    fill: How many numbers used in the model files (4 = name_XXXX.fits).
    data: The data for each component inside the model.
    extcorr: The self-extinction correction. If not carried out, saved as None.
    new: Whether or not the model was made with the newer version of collate.py.
    newIWall: The flux of an inner wall with a higher/lower altinh value.
    wallH: The inner wall height used by the look() function in plotting.
    filters: Filters used to calculate synthetic fluxes.
    synthFlux: Wavelengths and syntethic fluxes.

    METHODS
    __init__: Initializes an instance of the class, and loads in the relevant metadata.
    dataInit: Loads in the data to the object.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'.
    """

    def __init__(self, name, jobn, dpath=datapath, fill=3):
        """
        Initializes instances of this class and loads the relevant data into attributes.

        INPUTS
        name: Name of the object being modeled. Must match naming convention used for models.
        jobn: Job number corresponding to the model being loaded into the object. Again, must match convention.
        full_trans: BOOLEAN -- if 1 (True) will load data as a full or transitional disk. If 0 (False), as a pre-trans. disk.
        fill: How many numbers the input model file has (jobXXX vs. jobXXXX, etc.)
        """

        # Read in the fits file:
        stringnum       = str(jobn).zfill(fill)                         # Convert the jobn to the proper format
        fitsname        = dpath + name + '_' + stringnum + '.fits'      # Fits filename, preceeded by the path from paths section
        HDUlist         = fits.open(fitsname)                           # Opens the fits file for use
        header          = HDUlist[0].header                             # Stores the header in this variable

        # Initialize meta-data attributes for this object:
        self.name       = name
        self.jobn       = jobn
        self.mstar      = header['MSTAR']
        self.tstar      = header['TSTAR']
        self.rstar      = header['RSTAR']
        self.dist       = header['DISTANCE']
        self.mdot       = header['MDOT']
        self.alpha      = header['ALPHA']
        self.mui        = header['MUI']
        self.rdisk      = header['RDISK']
        self.amax       = header['AMAXS']
        self.eps        = header['EPS']
        self.tshock     = header['TSHOCK']
        self.temp       = header['TEMP']
        self.altinh     = header['ALTINH']
        self.wlcut_an   = header['WLCUT_AN']
        self.wlcut_sc   = header['WLCUT_SC']
        self.nsilcomp   = header['NSILCOMP']
        self.siltotab   = header['SILTOTAB']
        self.amorf_ol   = header['AMORF_OL']
        self.amorf_py   = header['AMORF_PY']
        self.forsteri   = header['FORSTERI']
        self.enstatit   = header['ENSTATIT']
        self.rin        = header['RIN']
        self.dpath      = dpath
        self.fill       = fill
        self.extcorr    = None
        self.filters    = {}
        self.synthFlux= {}
        try:
            self.mdotstar = header['MDOTSTAR']
        except KeyError:
            self.mdotstar = self.mdot

        HDUlist.close()
        return

    def dataInit(self, verbose=1):
        """
        Initialize data attributes for this object using nested dictionaries:
        wl is the wavelength (corresponding to all three flux arrays). Phot is the stellar photosphere emission.
        iWall is the flux from the inner wall. Disk is the emission from the angle file. Scatt is the scattered
        light emission. Loads in self-extinction array if available.

        INPUTS:
        verbose: BOOLEAN -- if 1 (True), will print out warnings about missing components.
        """

        stringnum    = str(self.jobn).zfill(self.fill)
        fitsname     = self.dpath + self.name + '_' + stringnum + '.fits'
        HDUdata      = fits.open(fitsname)
        header       = HDUdata[0].header

        # The new Python version of collate flips array indices, so must identify which collate.py was used:
        if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
            self.new = 1
        else:
            self.new = 0

        if self.new:
            # We will load in the components piecemeal based on the axes present in the header.
            # First though, we initialize with the wavelength array, since it's always present:
            self.data = {'wl': HDUdata[0].data[header['WLAXIS'],:]}

            # Now we can loop through the remaining possibilities:
            if 'PHOTAXIS' in header.keys():
                self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No photosphere data found for ' + self.name)
            if 'WALLAXIS' in header.keys():
                self.data['iwall'] = HDUdata[0].data[header['WALLAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No outer wall data found for ' + self.name)
            if 'ANGAXIS' in header.keys():
                self.data['disk'] = HDUdata[0].data[header['ANGAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No outer disk data found for ' + self.name)
            # Remaining components are not always (or almost always) present, so no warning given if missing!
            if 'SCATAXIS' in header.keys():
                self.data['scatt'] = HDUdata[0].data[header['SCATAXIS'],:]
                negScatt = np.where(self.data['scatt'] < 0.0)[0]
                if len(negScatt) > 0:
                    print('DATAINIT: WARNING: Some of your scattered light values are negative!')
            if 'EXTAXIS' in header.keys():
                self.extcorr       = HDUdata[0].data[header['EXTAXIS'],:]
        else:
            self.data = {'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'iwall': HDUdata[0].data[:,2], \
                         'disk': HDUdata[0].data[:,3]}

        HDUdata.close()
        return

    @keyErrHandle
    def calc_total(self, phot=1, wall=1, disk=1, dust=0, verbose=1, dust_fill=3, altinh=None, save=0, OTDpath=None):
        """
        Calculates the total flux for our object (likely to be used for plotting and/or analysis). Once calculated, it
        will be added to the data attribute for this object. If already calculated, will overwrite.
        
        INPUTS
        phot: BOOLEAN -- if 1 (True), will add photosphere component to the combined model.
        wall: BOOLEAN -- if 1 (True), will add inner wall component to the combined model.
        disk: BOOLEAN -- if 1 (True), will add disk component to the combined model.
        dust: INTEGER -- Must correspond to an opt. thin dust model number linked to a fits file in datapath directory.
        verbose: BOOLEAN -- if 1 (True), will print messages of what it's doing.
        dust_fill: INTEGER -- if 4, will look for a 4 digit valued dust file (i.e., name_OTD_XXXX.fits).
        altinh: FLOAT/INT -- if not None, will multiply inner wall flux by that amount.
        save: BOOLEAN -- if 1 (True), will print out the components to a .dat file.
        
        OUTPUT
        A boolean value, depending on whether it encountered any key errors or not.
        """
        
        # Add the components to the total flux, checking each component along the way:
        totFlux         = np.zeros(len(self.data['wl']), dtype=float)
        componentNumber = 1
        scatt           = 0     # For tracking if scattered light component exists
        
        if self.extcorr != None:
            componentNumber += 1
        if phot:
            if verbose:
                print('CALC_TOTAL: Adding photosphere component to the total flux.')
            totFlux     = totFlux + self.data['phot']
            componentNumber += 1
        if wall:
            if verbose:
                print('CALC_TOTAL: Adding inner wall component to the total flux.')
            if altinh != None:
                self.newIWall = self.data['iwall'] * altinh
                totFlux       = totFlux + self.newIWall     # Note: if save=1, will save iwall w/ the original altinh.
                self.wallH    = self.altinh * altinh
            else:
                totFlux       = totFlux + self.data['iwall']
                self.wallH    = self.altinh                 # Redundancy for plotting purposes.
                # If we tried changing altinh but want to now plot original, deleting the "newIWall" attribute from before.
                try:
                    del self.newIWall
                except AttributeError:
                    pass
            componentNumber += 1
        if disk:
            if verbose:
                print('CALC_TOTAL: Adding disk component to the total flux.')
            totFlux     = totFlux + self.data['disk']
            componentNumber += 1
        if dust != 0:
            dustNum     = str(dust).zfill(dust_fill)
            if OTDpath == None:
                dustHDU     = fits.open(self.dpath+self.name+'_OTD_'+dustNum+'.fits')
            else:
                dustHDU     = fits.open(OTDpath + self.name+'_OTD_'+dustNum+'.fits')
            if verbose:
                print('CALC_TOTAL: Adding optically thin dust component to total flux.')
            if self.new:
                self.data['dust']   = dustHDU[0].data[1,:]
            else:
                self.data['dust']   = dustHDU[0].data[:,1]
            totFlux     = totFlux + self.data['dust']
            componentNumber += 1
        
        # If scattered emission is in the dictionary, add it:
        if 'scatt' in self.data.keys():
            scatt       = 1
            if verbose:
                print('CALC_TOTAL: Adding scattered light component to the total flux.')
            totFlux     = totFlux + self.data['scatt']
            componentNumber += 1
        
        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print('CALC_TOTAL: Total flux calculated. Adding to the data structure.')
        self.data['total'] = totFlux
        componentNumber += 1
        
        # If save, create an output file with these components printed out:
        if save:
            outputTable = np.zeros([len(totFlux), componentNumber])

            # Populate the header and data table with the components and names:
            headerStr   = 'Wavelength, Total Flux, '
            outputTable[:, 0] = self.data['wl']
            outputTable[:, 1] = self.data['total']
            colNum      = 2
            if phot:
                headerStr += 'Photosphere, '
                outputTable[:, colNum] = self.data['phot']
                colNum += 1
            if wall:
                headerStr += 'Inner Wall, '
                outputTable[:, colNum] = self.data['iwall']
                colNum += 1
            if disk:
                headerStr += 'Outer Disk, '
                outputTable[:, colNum] = self.data['disk']
                colNum += 1
            if dust != 0:
                headerStr += 'Opt. Thin Dust, '
                outputTable[:, colNum] = self.data['dust']
                colNum += 1
            if scatt:
                headerStr += 'Scattered Light, '
                outputTable[:, colNum] = self.data['scatt']
                colNum += 1
            if self.extcorr != None:
                headerStr += 'Tau, '
                outputTable[:, colNum] = self.extcorr
            
            # Trim the header and save:
            headerStr  = headerStr[0:-2]
            filestring = '%s%s_%s.dat' % (self.dpath, self.name, str(self.jobn).zfill(self.fill))
            np.savetxt(filestring, outputTable, fmt='%.3e', delimiter=', ', header=headerStr, comments='#')
        
        return

    @keyErrHandle
    def calc_filters(self,filterspath=commonpath+'Filters/',obj=''):
        """
        Calculates the synthetic fluxes for our object at several typical photometric bands (likely
        to be used for chi square analysis). It reads the transmissivity of the filters of different instruments
        and then convolves the total emission of the disk with that transmissivty. Once calculated, it
        will be added to the synthFlux attribute for this object. If already calculated, will overwrite.

        INPUTS
        obj: TTS_Obs object, from which the observed photometric bands will be used. If not provided,
             synthethic fluces will be calculated for all photometric bands.

        OUTPUT
        A boolean value, depending on whether it encountered any key errors or not.

        NOTE:
        If synthetic fluxes want to be used when computing the chi square, these keywords
        need to be used when adding photometry:
        'CVSO', 'IRAC', 'MIPS', 'CANA', 'PACS', 'WISE', '2MASS'

        New instruments can be added with time. If any instrument is added,
        the instrKeylist should be updated here and in add_photometry.
        """
        
        #First check to see if there is a total component
        if 'total' not in self.data:
            raise ValueError("CALC_FILTERS: Model "+self.jobn+" does not have 'total' values, returning...")
            
        
        try:
            # If a TTS_Obs object is provided, it will calculate synthetic fluxes
            # for the instruments found in the object.
            instrKeylist = obj.photometry.keys()
        except:
            # If no TTS_Obs object is provided, it will calculate synthetic fluxes
            # for "all" instruments
            print('Calculating synthetic fluxes for all instruments and bands')
            instrKeylist = ['CVSO', 'IRAC','MIPS', 'CANA', 'PACS', 'WISE', '2MASS']

        for instrKey in instrKeylist:
            # For each instrument, it first reads the transmissivities and saves
            # them in the attribute filters as nested dictionaries.
            if instrKey == 'PACS': # Herschel
                PACS70filter         = np.loadtxt(filterspath+'PacsFilters/PACS_Tr70.dat')
                PACS160filter        = np.loadtxt(filterspath+'PacsFilters/PACS_Tr160.dat')
                self.filters[instrKey] = {'f70.0': {'wl':PACS70filter[:,0],'trans':PACS70filter[:,1]},\
                                        'f160.0': {'wl':PACS160filter[:,0],'trans':PACS160filter[:,1]}}
            elif instrKey == 'IRAC': # Spitzer
                I1filter             = np.loadtxt(filterspath+'filtrosSpitzer/iractr1.dat')
                I2filter             = np.loadtxt(filterspath+'filtrosSpitzer/iractr2.dat')
                I3filter             = np.loadtxt(filterspath+'filtrosSpitzer/iractr3.dat')
                I4filter             = np.loadtxt(filterspath+'filtrosSpitzer/iractr4.dat')
                self.filters[instrKey] = {'f3.6': {'wl':I1filter[:,0],'trans':I1filter[:,1]},\
                                        'f4.5': {'wl':I2filter[:,0],'trans':I2filter[:,1]},\
                                        'f5.8': {'wl':I3filter[:,0],'trans':I3filter[:,1]} ,\
                                        'f8.0': {'wl':I4filter[:,0],'trans':I4filter[:,1]}}
            elif instrKey == 'MIPS': #Spitzer
                MIPSfilter           = np.loadtxt(filterspath+'filtrosSpitzer/mipstr1.dat')
                self.filters[instrKey] = {'f24.0': {'wl':MIPSfilter[:,0],'trans':MIPSfilter[:,1]}}
            elif instrKey == '2MASS':
                J2MASSfilter      = np.loadtxt(filterspath+'2MASSfilters/2MASSJ.dat')
                H2MASSfilter      = np.loadtxt(filterspath+'2MASSfilters/2MASSH.dat')
                K2MASSfilter      = np.loadtxt(filterspath+'2MASSfilters/2MASSKs.dat')
                self.filters[instrKey] = {'f1.235': {'wl':J2MASSfilter[:,0]/10e3,'trans':J2MASSfilter[:,1]},\
                                        'f1.662': {'wl':H2MASSfilter[:,0]/10e3,'trans':H2MASSfilter[:,1]},\
                                        'f2.159': {'wl':K2MASSfilter[:,0]/10e3,'trans':K2MASSfilter[:,1]}}
            elif instrKey == 'WISE':
                W1filter      = np.loadtxt(filterspath+'WISE_filters/RSR-W1.txt')
                W2filter      = np.loadtxt(filterspath+'WISE_filters/RSR-W2.txt')
                W3filter      = np.loadtxt(filterspath+'WISE_filters/RSR-W3.txt')
                W4filter      = np.loadtxt(filterspath+'WISE_filters/RSR-W4.txt')
                self.filters[instrKey] = {'f3.4': {'wl':W1filter[:,0],'trans':W1filter[:,1]},\
                                        'f4.6': {'wl':W2filter[:,0],'trans':W2filter[:,1]},\
                                        'f12.0': {'wl':W3filter[:,0],'trans':W3filter[:,1]},\
                                        'f22.0': {'wl':W4filter[:,0],'trans':W4filter[:,1]}}
            elif instrKey == 'CVSO': # CIDA Variability Survey
                VCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_V-1_km.txt')
                RCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_R-1_km.txt')
                ICVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_I-1_km.txt')
                self.filters[instrKey] = {'f0.55': {'wl':VCVSOfilter[:,0]*0.001,'trans':VCVSOfilter[:,1]},\
                                        'f0.64': {'wl':RCVSOfilter[:,0]*0.001,'trans':RCVSOfilter[:,1]},\
                                        'f0.79': {'wl':ICVSOfilter[:,0]*0.001,'trans':ICVSOfilter[:,1]}}
            elif instrKey == 'CANA': # CanaryCam
                Si2filter        = np.loadtxt(filterspath+'CanariCam_filters/Si2.txt')
                Si4filter        = np.loadtxt(filterspath+'CanariCam_filters/Si4.txt')
                Si5filter        = np.loadtxt(filterspath+'CanariCam_filters/Si-5.txt')
                Si6filter        = np.loadtxt(filterspath+'CanariCam_filters/Si-6.txt')
                self.filters[instrKey] = {'f8.7': {'wl':Si2filter[:,0],'trans':Si2filter[:,1]},\
                                        'f10.3': {'wl':Si4filter[:,0],'trans':Si4filter[:,1]},\
                                        'f11.6': {'wl':Si5filter[:,0],'trans':Si5filter[:,1]},\
                                        'f12.5': {'wl':Si6filter[:,0],'trans':Si6filter[:,1]}}
            else:
                print('No transmissivities found for instrument '+instrKey)
                continue

            intFlux = []
            wlcfilter = []
            for bandkey in self.filters[instrKey].keys():
                # For each band for the given instrument
                # First it interpolates the model at the wavelengths for which
                # we have values of the transmissivity
                Fmodatfilter = np.interp(self.filters[instrKey][bandkey]['wl'],\
                                         self.data['wl'], self.data['total'])

                # Now it integrates and convolves the model with the transmissivity
                # of that particular band. It uses a trapezoidal integration.
                s1 = np.trapz(Fmodatfilter*self.filters[instrKey][bandkey]['trans'],\
                              x=self.filters[instrKey][bandkey]['wl'])
                # It integrates the transmissivity to normalize the convolved flux
                s2 = np.trapz(self.filters[instrKey][bandkey]['trans'],\
                              x=self.filters[instrKey][bandkey]['wl'])

                intF = s1 / s2

                wlcfilter.append(float(bandkey[1:])) # Central wavelength of band (microns)
                intFlux.append(intF) # Synthetic flux at the band

            # We convert the two resulting lists in arrays and sort them
            wlcfilter = np.array(wlcfilter)
            sortindex = np.argsort(wlcfilter)
            wlcfilter = wlcfilter[sortindex]
            intFlux = np.array(intFlux)[sortindex]

            # And it saves the central wavelengths and synthetic fluxes of all
            # the bands of the instrument in the synthFlux attribute as a
            # nested dictionary
            self.synthFlux[instrKey] = {'wl':wlcfilter,'lFl':intFlux}

        return

    def blueExcessModel(self, shockPath=shockpath, veilVal=None, Vflux=None):
        """
        Adding the excess emission in the optical and near-UV from accretion shock models to the total emission. The
        models are taken from Laura Ingleby's models.

        NOTE1: For this to work, you need to set phot=0 during the calculation of the total model component!

        NOTE2: This section is not very generalized, and needs work. - Dan

        INPUT
        shockPath: Where the accretion shock model data is located.
        """

        print('WARNING: WORKS, NOT GENERALIZED!!!')

        # Start by loading in the shock model table:
        if self.name.endswith('pt'):
            shockTable = np.loadtxt(shockPath+self.name[:-2]+'.dat', skiprows=1)
            shockLong  = np.loadtxt(shockPath+'shock_'+self.name[:-2]+'.dat')
            self.data['shockLong'] = {'wl': shockLong[:,0]*1e-4, 'lFl': shockLong[:,1]*shockLong[:,0]}
        else:
            shockTable = np.loadtxt(shockPath+self.name+'.dat', skiprows=1)
            shockLong  = np.loadtxt(shockPath+'shock_'+self.name+'.dat')
            self.data['shockLong'] = {'wl': shockLong[:,0]*1e-4, 'lFl': shockLong[:,1]*shockLong[:,0]}

        # Convert everything to the correct units:
        shockTable[:,1] *= shockTable[:,0]      # Make the flux be in erg s-1 cm-2
        shockTable[:,2] *= shockTable[:,0]      # Same units for WTTS component
        shockTable[:,3] *= shockTable[:,0]      # Same units for shock component
        shockTable[:,0] *= 1e-4                 # Wavelength in microns

        # Check if any bad (super low) data points in WTTS component, and then remove them:
        lowVals = np.where(shockTable[:,2] < 1e-15)[0]
        if len(lowVals) != 0:
            shockTable = np.delete(shockTable, lowVals, 0)

        # Check for NaNs in the data, and if they exist, remove them:
        if np.isnan(np.sum(shockTable[:,2])):
            badVals    = np.where(np.isnan(shockTable[:,2]))
            shockTable = np.delete(shockTable, badVals, 0)
        if np.isnan(np.sum(shockTable[:,3])):
            badVals2   = np.where(np.isnan(shockTable[:,3]))
            shockTable = np.delete(shockTable, badVals2, 0)

        # Define and add the accretion shock data:
        shockWL        = shockTable[:,0].copy() # I might make cuts later, so want to copy now
        shockFlux      = shockTable[:,1].copy()
        shockMod       = shockTable[:,2].copy()
        self.data['shock']  = {'wl': shockWL, 'lFl': shockFlux}
        self.data['WTTS']   = {'wl': shockWL, 'lFl': shockMod}
        if veilVal is not None:
            normVfactor     = Vflux / (1 + veilVal)
            self.data['WTTS']['lFl'] = normalize(self.data['WTTS'], 0.545, normVfactor)


        # Need to interpolate the model onto the appropriate wavelength grid:
        wlgrid = np.where(np.logical_and(self.data['wl'] <= shockTable[-1,0], self.data['wl'] >= shockTable[0,0]))[0]
        totalInterp    = np.interp(shockTable[:,0], self.data['wl'][wlgrid], self.data['total'][wlgrid])
        # Now, take the wlgrid out of the original arrays:
        self.data['total'][wlgrid] = np.nan

        # Add to the total data, and then plop back into the full grid.
        excessTotal        = totalInterp + self.data['WTTS']['lFl'] + shockTable[:,3]
        oldWavelength      = self.data['wl'].copy()     # Save a copy for later
        self.data['wl']    = np.append(self.data['wl'], shockTable[:,0])
        self.data['total'] = np.append(self.data['total'], excessTotal)
        sortInd = np.argsort(self.data['wl'])
        self.data['wl']    = self.data['wl'][sortInd]
        self.data['total'] = self.data['total'][sortInd]

        # Now I need to add the shock model for past 1 micron...sigh. Sorry for the weird repetition.
        wlgrid2 = np.where(np.logical_and(self.data['wl'] <= shockLong[-1,0]*1e-4, self.data['wl'] >= shockTable[-1,0]))[0]
        secondInterp   = np.interp(self.data['wl'][wlgrid2], shockLong[:,0]*1e-4, shockLong[:,1]*shockLong[:,0])
        self.data['total'][wlgrid2] += secondInterp

        # Doublecheck for the existence of NaNs in your model:
        #if np.isnan(np.sum(self.data['total'])):
        #    print('BLUEEXCESSMODEL: WARNING! There are NaNs in the total component of your model.')

        # Now all of the other components are not on the same grid. Let's interpolate all of them:
        for key in self.data.keys():
            if key == 'total' or key == 'wl' or key == 'shock' or key == 'WTTS' or key == 'shockLong':
                pass
            else:
                self.data[key] = np.interp(self.data['wl'], oldWavelength, self.data[key])
        try:
            self.newIWall = np.interp(self.data['wl'], oldWavelength, self.newIWall)
        except AttributeError:
            pass
        try:
            self.newOWall = np.interp(self.data['wl'], oldWavelength, self.newOWall)
        except AttributeError:
            pass

        # Normalize the Kenyon and Hartmann photosphere to the WTTS photosphere for rough consistency:
        normFactor = np.max(self.data['WTTS']['lFl'][-100:])
        photAnchor = self.data['phot'][np.where(self.data['wl'] == shockTable[-1,0])[0]]
        self.data['phot'] *= normFactor / photAnchor

        # If we use this model, we need to add the KH photosphere for wavelengths greater than a micron:
        WTTS_ind = np.where(self.data['wl'] > shockTable[-1,0])[0]
        self.data['total'][WTTS_ind] += self.data['phot'][WTTS_ind]

        return

class PTD_Model(TTS_Model):
    """
    Contains all the data and meta-data for a PTD Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.

    ATTRIBUTES
    name: Name of the object (e.g., CVSO109, V410Xray-2, ZZ_Tau, etc.).
    jobn: The job number corresponding to this model.
    mstar: Star's mass.
    tstar: Star's effective temperature, based on Kenyon and Hartmann 1995.
    rstar: Star's radius.
    dist: Distance to the star.
    mdot: Mass accretion rate.
    alpha: Alpha parameter (from the viscous alpha disk model).
    mui: Inclination of the system.
    rdisk: The outer radius of the disk.
    amax: The "maximum" grain size in the disk. (or just suspended in the photosphere of the disk?)
    eps: The epsilon parameter, i.e., the amount of dust settling in the disk.
    tshock: The temperature of the shock at the stellar photosphere.
    temp: The temperature at the outer wall component of the model.
    itemp: The temperature of the inner wall component of the model.
    altinh: Scale heights of extent of the inner wall.
    wlcut_an:
    wlcut_sc:
    nsilcomp: Number of silicate compounds.
    siltotab: Total silicate abundance.
    amorf_ol:
    amorf_py:
    forsteri: Forsterite Fractional abundance.
    enstatit: Enstatite Fractional abundance.
    rin: The inner radius in AU.
    dpath: Path where the data files are located.
    fill: How many numbers used in the model files (4 = name_XXXX.fits).
    data: The data for each component inside the model.
    extcorr: The self-extinction correction. If not carried out, saved as None.
    new: Whether or not the model was made with the newer version of collate.py.
    newIWall: The flux of an inner wall with a higher/lower altinh value.
    newOWall: The flux of an outer wall with a higher/lower altinh value.
    iwallH: The inner wall height used by the look() function in plotting.
    wallH: The outer wall height used by the look() function in plotting.

    METHODS
    __init__: initializes an instance of the class, and loads in the relevant metadata. No change.
    dataInit: Loads in the relevant data to the object. This differs from that of TTS_Model.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'. This also differs from TTS_Model.
    """

    def dataInit(self, altname=None, jobw=None, fillWall=3, wallpath = '', verbose =1, **searchKwargs):
        """
        Initialize data attributes for this object using nested dictionaries:
        wl is the wavelength (corresponding to all three flux arrays). Phot is the stellar photosphere emission.
        iwall is the flux from the inner wall. Disk is the emission from the angle file. owall is the flux from
        the outer wall. Scatt is the scattered light emission. Also adds self-extinction array if available.

        You should either supply the job number of the inner wall file, or the kwargs used to find it via a
        search. Jobw

        INPUTS
        altname: An alternate name for the inner wall file if necessary.
        jobw: The job number of the wall. Can be a string of 'XXX' or 'XXXX' based on the filename, or just the integer.
        fillWall: How many numbers used in the model file for the inner wall (4 = name_XXXX.fits).
        wallpath: Path to the wall files, otherwise use the default self.dpath
        **searchkwargs: Kwargs corresponding to parameters in the header that can be used to find the jobw value if
                        you don't already know it. Otherwise, not necessary for the function call.


        CURRENT HACKED MODIFICATIONS:
            Added ability to have an inner disk along with an inner wall

        """

        if jobw == None and len(searchKwargs) == 0:
            raise IOError('DATAINIT: You must enter either a job number or kwargs to match or search for an inner wall.')

        if jobw != None:
            # If jobw is an integer, make into a string:
            jobw          = str(jobw).zfill(fillWall)

            # The case in which you supplied the job number of the inner wall:
            if altname == None:
                if wallpath == None:
                    fitsname  = self.dpath + self.name + '_' + jobw + '.fits'
                if wallpath != None:
                    fitsname  = wallpath + self.name + '_' + jobw + '.fits'

                HDUwall   = fits.open(fitsname)
            else:
                if wallpath == None:
                    fitsname  = self.dpath + altname + '_' + jobw + '.fits'
                if wallpath != None:
                    fitsname  = wallpath + altname + '_' + jobw + '.fits'

                HDUwall   = fits.open(fitsname)

            # Make sure the inner wall job you supplied is, in fact, an inner wall.
            if verbose:
                if 'NOEXT' not in HDUwall[0].header.keys():
                    #raise IOError('DATAINIT: Job you supplied is not an inner wall or needs to be collated again!')
                    print('DATAINIT: Job you supplied is not ONLY an inner wall and may need to be collated again!')


            # Now, load in the disk data:
            stringNum     = str(self.jobn).zfill(self.fill)
            HDUdata       = fits.open(self.dpath + self.name + '_' + stringNum + '.fits')
            header        = HDUdata[0].header

            # Check if it's an old version or a new version:
            if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
                self.new  = 1
            else:
                self.new  = 0

            # Define the inner wall height.
            self.iwallH   = HDUwall[0].header['ALTINH']
            self.itemp    = HDUwall[0].header['TEMP']
            self.ijobn    = HDUwall[0].header['JOBNUM']




            # Depending on old or new version is how we will load in the data. We require the wall be "new":
            if self.new:
                # Correct for self extinction:
                try:
                    iwallFcorr= HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])
                except KeyError:
                    print('DATAINIT: WARNING! No extinction correction can be made for job ' + str(self.jobn)+'!')

                    iwallFcorr= HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]

                # We will load in the components piecemeal based on the axes present in the header.
                # First though, we initialize with the wavelength and wall, since they're always present.

                #If there is an inner disk, add that as well.
                if 'NOEXT' not in HDUwall[0].header.keys():
                    try:
                        idiskFcorr= HDUwall[0].data[HDUwall[0].header['ANGAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])
                    except KeyError:
                        idiskFcorr= HDUwall[0].data[HDUwall[0].header['ANGAXIS'],:]

                    self.data = {'wl': HDUdata[0].data[header['WLAXIS'],:], 'iwall': iwallFcorr, 'idisk': idiskFcorr}

                    #Add information about the disk
                    self.ialpha      = HDUwall[0].header['ALPHA']
                    self.irdisk      = HDUwall[0].header['RDISK']
                    self.iamax       = HDUwall[0].header['AMAXS']
                    self.ieps        = HDUwall[0].header['EPS']
                    self.insilcomp   = HDUwall[0].header['NSILCOMP']
                    self.isiltotab   = HDUwall[0].header['SILTOTAB']
                    self.iamorf_ol   = HDUwall[0].header['AMORF_OL']
                    self.iamorf_py   = HDUwall[0].header['AMORF_PY']
                    self.iforsteri   = HDUwall[0].header['FORSTERI']
                    self.ienstatit   = HDUwall[0].header['ENSTATIT']
                    self.irin        = HDUwall[0].header['RIN']

                else:
                    self.data = {'wl': HDUdata[0].data[header['WLAXIS'],:], 'iwall': iwallFcorr}

                # Now we can loop through the remaining possibilities:
                if 'PHOTAXIS' in header.keys():
                    self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
                else:
                    print('DATAINIT: Warning: No photosphere data found for ' + self.name)
                if 'WALLAXIS' in header.keys():
                    self.data['owall']= HDUdata[0].data[header['WALLAXIS'],:]
                else:
                    print('DATAINIT: Warning: No outer wall data found for ' + self.name)
                if 'ANGAXIS' in header.keys():
                    self.data['odisk'] = HDUdata[0].data[header['ANGAXIS'],:]
                else:
                    print('DATAINIT: Warning: No outer disk data found for ' + self.name)
                # Remaining components are not always (or almost always) present, so no warning given if missing!
                if 'SCATAXIS' in header.keys():
                    self.data['scatt']= HDUdata[0].data[header['SCATAXIS'],:]
                    negScatt = np.where(self.data['scatt'] < 0.0)[0]
                    if len(negScatt) > 0:
                        print('DATAINIT: WARNING: Some of your scattered light values are negative!')
                if 'EXTAXIS' in header.keys():
                    self.extcorr      = HDUdata[0].data[header['EXTAXIS'],:]
            else:
                self.data = ({'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'owall': HDUdata[0].data[:,2],
                              'disk': HDUdata[0].data[:,3], 'iwall': HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]})

        else:
            # When doing the searchJobs() call, use **searchKwargs to pass that as the keyword arguments to searchJobs!
            if altname == None:
                match = searchJobs(self.name, dpath=self.dpath, **searchKwargs)
            else:
                match = searchJobs(altname, dpath=self.dpath, **searchKwargs)
            if len(match) == 0:
                raise IOError('DATAINIT: No inner wall model matches these parameters!')
            elif len(match) > 1:
                raise IOError('DATAINIT: Multiple inner wall models match. Do not know which one to pick.')
            else:
                if altname == None:
                    fitsname = self.dpath + self.name + '_' + match[0] + '.fits'
                else:
                    fitsname = self.dpath + altname + '_' + match[0] + '.fits'
                HDUwall  = fits.open(fitsname)

                # Make sure the inner wall job you supplied is, in fact, an inner wall.
                if 'NOEXT' not in HDUwall[0].header.keys():
                    raise IOError('DATAINIT: Job found is not an inner wall or needs to be collated again!')

                # Now, load in the disk data:
                stringNum    = str(self.jobn).zfill(self.fill)
                HDUdata      = fits.open(self.dpath + self.name + '_' + stringNum + '.fits')
                header       = HDUdata[0].header

                # Check if it's an old version or a new version:
                if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
                    self.new = 1
                else:
                    self.new = 0

                # Define the inner wall height:
                self.iwallH  = HDUwall[0].header['ALTINH']
                self.itemp   = HDUwall[0].header['TEMP']

                # Depending on old or new version is how we will load in the data. We require the wall be "new":
                if self.new:
                    # Correct for self extinction:
                    iwallFcorr = HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])

                    # We will load in the components piecemeal based on the axes present in the header.
                    # First though, we initialize with the wavelength and wall, since they're always present:
                    self.data  = {'wl': HDUdata[0].data[header['WLAXIS'],:], 'iwall': iwallFcorr}

                    # Now we can loop through the remaining possibilities:
                    if 'PHOTAXIS' in header.keys():
                        self.data['phot'] = HDUdata[0].data[header['PHOTAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No photosphere data found for ' + self.name)
                    if 'WALLAXIS' in header.keys():
                        self.data['owall']= HDUdata[0].data[header['WALLAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No outer wall data found for ' + self.name)
                    if 'ANGAXIS' in header.keys():
                        self.data['disk'] = HDUdata[0].data[header['ANGAXIS'],:]
                    else:
                        print('DATAINIT: Warning: No outer disk data found for ' + self.name)
                    # Remaining components are not always (or almost always) present, so no warning given if missing!
                    if 'SCATAXIS' in header.keys():
                        self.data['scatt']= HDUdata[0].data[header['SCATAXIS'],:]
                        negScatt = np.where(self.data['scatt'] < 0.0)[0]
                        if len(negScatt) > 0:
                            print('DATAINIT: WARNING: Some of your scattered light values are negative!')
                    if 'EXTAXIS' in header.keys():
                        self.extcorr      = HDUdata[0].data[header['EXTAXIS'],:]
                else:
                    self.data = ({'wl': HDUdata[0].data[:,0], 'phot': HDUdata[0].data[:,1], 'owall': HDUdata[0].data[:,2],
                                  'disk': HDUdata[0].data[:,3], 'iwall': HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]})
        HDUdata.close()
        return

    @keyErrHandle
    def calc_total(self, phot=1, iwall=1, idisk=1, owall=1, odisk = 1, dust=0, verbose=1, dust_fill=3, altInner=None, altOuter=None, save=0, OTDpath=None):
        """
        Calculates the total flux for our object (likely to be used for plotting and/or analysis). Once calculated, it
        will be added to the data attribute for this object. If already calculated, will overwrite.

        INPUTS
        phot: BOOLEAN -- if 1 (True), will add photosphere component to the combined model.
        wall: BOOLEAN -- if 1 (True), will add inner wall component to the combined model.
        disk: BOOLEAN -- if 1 (True), will add disk component to the combined model.
        owall: BOOLEAN -- if 1 (True), will add outer wall component to the combined model.
        dust: INTEGER -- Must correspond to an opt. thin dust model number linked to a fits file in datapath directory.
        verbose: BOOLEAN -- if 1 (True), will print messages of what it's doing.
        dust_fill: INT -- How many numbers used in the model file for the optically thin dust (4 = name_OTD_XXXX.fits).
        altInner: FLOAT/INT -- if not None, will multiply inner wall flux by that amount.
        altOuter: FLOAT/INT -- if not None, will multiply outer wall flux by that amount.
        save: BOOLEAN -- if 1 (True), will print out the components to a .dat file.
        OTDpath: STRING -- optional path to OTD files
        """

        # Add the components to the total flux, checking each component along the way:
        totFlux         = np.zeros(len(self.data['wl']), dtype=float)
        componentNumber = 1
        if phot:
            if verbose:
                print('CALC_TOTAL: Adding photosphere component to the total flux.')
            totFlux     = totFlux + self.data['phot']
            componentNumber += 1
        if iwall:
            if verbose:
                print('CALC_TOTAL: Adding inner wall component to the total flux.')
            if altInner != None:
                self.newIWall = self.data['iwall'] * altInner
                totFlux       = totFlux + self.newIWall     # Note: if save=1, will save iwall w/ the original altinh.
                self.wallH    = self.iwallH * altInner
            else:
                totFlux = totFlux + self.data['iwall']
                self.wallH    = self.iwallH                 # Redundancy for plotting purposes.
                # If we tried changing altinh but want to now plot original, deleting the "newIWall" attribute from before.
                try:
                    del self.newIWall
                except AttributeError:
                    pass
            componentNumber += 1

        if idisk:
            if verbose:
                print('CALC_TOTAL: Adding inner disk component to the total flux.')
            totFlux     = totFlux + self.data['idisk']
            componentNumber += 1

        if odisk:
            if verbose:
                print('CALC_TOTAL: Adding outer disk component to the total flux.')
            totFlux     = totFlux + self.data['odisk']
            componentNumber += 1

        if owall:
            if verbose:
                print('CALC_TOTAL: Adding outer wall component to the total flux.')
            if altOuter != None:
                self.newOWall = self.data['owall'] * altOuter
                totFlux = totFlux + self.newOWall           # Note: if save=1, will save owall w/ the original altinh.
                self.owallH   = self.altinh * altOuter
            else:
                totFlux       = totFlux + self.data['owall']
                self.owallH   = self.altinh
                # If we tried changing altinh but want to now plot original, deleting the "newOWall" attribute from before.
                try:
                    del self.newOWall
                except AttributeError:
                    pass
            componentNumber += 1
        if dust != 0:
            dustNum     = str(dust).zfill(dust_fill)
            if OTDpath == None:
                dustHDU     = fits.open(self.dpath+self.name+'_OTD_'+dustNum+'.fits')
            else:
                dustHDU     = fits.open(OTDpath + self.name+'_OTD_'+dustNum+'.fits')
            if verbose:
                print('CALC_TOTAL: Adding optically thin dust component to total flux.')
            if self.new:
                self.data['dust'] = dustHDU[0].data[1,:]
            else:
                self.data['dust'] = dustHDU[0].data[:,1]
            totFlux     = totFlux + self.data['dust']
            componentNumber += 1

        # If scattered emission is in the dictionary, add it:
        if 'scatt' in self.data.keys():
            scatt       = 1
            if verbose:
                print('CALC_TOTAL: Adding scattered light component to the total flux.')
            totFlux     = totFlux + self.data['scatt']
            componentNumber += 1

        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print('CALC_TOTAL: Total flux calculated. Adding to the data structure.')
        self.data['total'] = totFlux
        componentNumber += 1


        # If save, create an output file with these components printed out:
        if save:
            outputTable = np.zeros([len(totFlux), componentNumber])

            # Populate the header and data table with the components and names:
            headerStr   = 'Wavelength, Total Flux, '
            outputTable[:, 0] = self.data['wl']
            outputTable[:, 1] = self.data['total']
            colNum      = 2
            if phot:
                headerStr += 'Photosphere, '
                outputTable[:, colNum] = self.data['phot']
                colNum += 1
            if wall:
                headerStr += 'Inner Wall, '
                outputTable[:, colNum] = self.data['iwall']
                colNum += 1
            if owall:
                headerStr += 'Outer Wall, '
                outputTable[:, colNum] = self.data['owall']
                colNum += 1
            if disk:
                headerStr += 'Outer Disk, '
                outputTable[:, colNum] = self.data['disk']
                colNum += 1
            if dust != 0:
                headerStr += 'Opt. Thin Dust, '
                outputTable[:, colNum] = self.data['dust']
                colNum += 1
            if scatt:
                headerStr += 'Scattered Light, '
                outputTable[:, colNum] = self.data['scatt']
                colNum += 1

            # Trim the header and save:
            headerStr  = headerStr[0:-2]
            filestring = '%s%s_%s.dat' % (self.dpath, self.name, str(self.jobn).zfill(self.fill))
            np.savetxt(filestring, outputTable, fmt='%.3e', delimiter=', ', header=headerStr, comments='#')

        return

class TTS_Obs(object):
    """
    Contains all the observational data for a given target system. Allows you to create a pickle with the data, so it can
    be reloaded in at a future time without the need to re-initialize the object. However, to open up the pickle, you will
    need to have this source code where Python can access it.

    ATTRIBUTES
    name: The name of the target whose observations this represents.
    spectra: The spectra measurements for said target.
    photometry: The photometry measurements for said target.
    ulim: Which (if any) photometry points are upper limits.
    phot_dens: Density (per wavelength) of photometric points.
    spec_dens: Density (per wavelength) of points in each spectrum.

    METHODS
    __init__: Initializes an instance of this class. Creates initial attributes (name and empty data dictionaries).
    add_spectra: Adds an entry (or replaces an entry) in the spectra attribute dictionary.
    add_photometry: Adds an entry (or replaces an entry) in the photometry attribute dictionary.
    SPPickle: Saves the object as a pickle to be reloaded later. This will not work if you've reloaded the module before saving.
    """

    def __init__(self, name):
        """
        Initializes instances of the class and loads in data to the proper attributes.

        INPUTS
        name: The name of the target for which the data represents.
        """
        # Initalize attributes as empty. Can add to the data later.
        self.name       = name
        self.spectra    = {}
        self.photometry = {}
        self.ulim       = []
        self.spec_dens = {}
        self.phot_dens = 0.0

    def add_spectra(self, scope, wlarr, fluxarr, errors=None, py2 = False):
        """
        Adds an entry to the spectra attribute.

        INPUTS
        scope: The telescope or instrument that the spectrum was taken with.
        wlarr: The wavelenth array of the data. Should be in microns. Note: this is not checked.
        fluxarr: The flux array of the data. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        """

        # Check if the telescope data already exists in the data file:
        if scope in self.spectra.keys():
            print('ADD_SPECTRA: Warning! This will overwrite current entry!')
            tries               = 1
            while tries <= 5:                                           # Give user 5 chances to choose if overwrite data or not
                if py2:
                    proceed         = raw_input('Proceed? (Y/N): ')         # Prompt and collect manual answer - requires Y,N,Yes,No (not case sensitive)
                else:
                    proceed         = input('Proceed? (Y/N): ')
                if proceed.upper() == 'Y' or proceed.upper() == 'YES':  # If Y or Yes, overwrite file, then break out of loop
                    print('ADD_SPECTRA: Replacing entry.')
                    if errors == None:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
                    else:
                        self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO': # If N or No, do not overwrite data and return
                    print('ADD_SPECTRA: Will not replace entry. Returning now.')
                    return
                else:
                    tries       = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if errors == None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
            else:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
        return

    def add_photometry(self, scope, wlarr, fluxarr, errors=None, ulim=0, verbose = 1, py2 = False):
        """
        Adds an entry to the photometry attribute.

        INPUTS
        scope: The telescope or instrument that the photometry was taken with.
        wlarr: The wavelength array of the data. Can also just be one value if an individual point. Should be in microns. Note: this is not checked.
        fluxarr: The flux array corresponding to the data. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        ulim: BOOLEAN -- whether or not this photometric data is or is not an upper limit.

        NOTE:
        If synthetic fluxes want to be used when computing the chi square, these keywords
        need to be used when adding photometry:
        'CVSO', 'IRAC', 'MIPS', 'CANA', 'PACS', 'WISE', '2MASS'

        New instruments can be added with time. If any instrument is added,
        the instrKeylist should be updated.
        """
        # Check if scope is a string, and make it upper case
        # so that it matches filters keywords
        if type(scope) == str:
            scope.upper()
        else:
            raise IOError('scope should be a string.')

        # Check if instrument is in list of instruments with available synthetic fluxes
        instrKeylist = ['CVSO', 'IRAC','MIPS', 'CANA', 'PACS', 'WISE', '2MASS']
        if scope not in instrKeylist:
            if verbose:
                print('Synthetic fluxes for instrument "'+scope+'" cannot be currently computed.')
                print('Available instruments are: '+', '.join(instrKeylist))

        # Check that the wavelengths and fluxes entered make sense
        try: # If values entered are numeric, make them an array
            wlarr[0]
        except:
            try: # In case entered fluxes and wavelengths are of strange types
                wlarr = np.array([wlarr])
                fluxarr = np.array([fluxarr])
            except:
                raise IOError('ADD_PHOTOMETRY: The fluxes and wavelengths should be a list, an array, or individual numeric values.')
        if type(wlarr) == list: # If values entered are a list, make them an array
            wlarr = np.array(wlarr)
        if type(fluxarr) == list:
            fluxarr = np.array(fluxarr)
        if type(wlarr) == str or type(fluxarr) == str: # Check if they are strings
            raise IOError('ADD_PHOTOMETRY: The flux and wavelength should NOT be strings')
        if len(wlarr) != len(fluxarr): # Check if they have the same length
            raise IOError('ADD_PHOTOMETRY: The flux and wavelength should have the same number of elements')

        # Check if the telescope data already exists in the data file:
        if scope in self.photometry.keys():
            print('ADD_PHOTOMETRY: Warning! This will overwrite current entry!')
            tries                   = 1
            while tries <= 5:                                               # Give user 5 chances to choose if overwrite data or not
                if py2:
                    proceed             = raw_input('Proceed? (Y/N): ')         # Prompt and collect manual answer - requires Y,N,Yes,No (not case sensitive)
                else:
                    proceed             = input('Proceed? (Y/N): ')
                if proceed.upper() == 'Y' or proceed.upper() == 'YES':      # If Y or Yes, overwrite file, then break out of loop
                    print('ADD_PHOTOMETRY: Replacing entry.')
                    if errors == None:
                        self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}
                    else:
                        self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
                    if ulim == 1:
                        self.ulim.append(scope)                             # If upper limit, append metadata to ulim attribute list.
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO':     # If N or No, do not overwrite data and return
                    print('ADD_PHOTOMETRY: Will not replace entry. Returning now.')
                    return
                else:
                    tries           = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if errors == None:
                self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}     # If not an overwrite, writes data to the object's photometry attribute dictionary.
            else:
                self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
            if ulim == 1:
                self.ulim.append(scope)                                     # If upper limit, append metadata to ulim attribute list.
        # We reset the attribute phot_dens, since with new photometry it should be recalculated
        self.phot_dens = 0.0
        return

    def SPPickle(self, picklepath, clob = False, fill = 3):
        """
        Saves the object as a pickle. Damn it Jim, I'm a doctor not a pickle farmer!

        WARNING: If you reload the module BEFORE you save the observations as a pickle, this will NOT work! I'm not
        sure how to go about fixing this issue, so just be aware of this.

        INPUTS
        picklepath: The path where you will save the pickle. I recommend datapath for simplicity.
        clob: boolean, if set to True, will clobber the old pickle
        fill: How many numbers used in the model files (4 = name_XXXX.fits).

        OUTPUT:
        A pickle file of the name [self.name]_obs.pkl in the directory provided in picklepath.
        """

        # Check whether or not the pickle already exists:
        pathlist        = filelist(picklepath)
        outname         = self.name + '_obs.pkl'
        count           = 1
        while 1:
            if outname in pathlist and clob == False:
                if count == 1:
                    print('SPPICKLE: Pickle already exists in directory. For safety, will change name.')
                countstr= str(count).zfill(fill)
                count   = count + 1
                outname = self.name + '_obs_' + countstr + '.pkl'
            else:
                break
        # Now that that's settled, let's save the pickle.
        f               = open(picklepath + outname, 'wb')
        cPickle.dump(self, f)
        f.close()
        return

class Red_Obs(TTS_Obs):
    """
    A similar class to TTS_Obs, except meant to be utilized for observations that have not yet been
    dereddened. Once dereddened, the pickle will be saved as a TTS_Obs object. If saved prior to
    dereddening, the pickle will be associated with Red_Obs instead. I recommend keeping both.

    """

    def dered(self, Av, Av_unc, law, picklepath, flux=1, lpath=commonpath, err_prop=1, UV = 0, clob = False):
        """
        Deredden the spectra/photometry present in the object, and then convert to TTS_Obs structure.
        This function is adapted from the IDL procedure 'dered_calc.pro' (written by Melissa McClure).
        This requires the spectral fluxes to be units of erg s-1 cm-2 cm-1.

        INPUTS
        Av: The Av extinction value.
        Av_unc: The uncertainty in the Av value provided.
        law: The extinction law to be used -- these extinction laws are found in the ext_laws.pkl file.
             The options you have are 'mkm09_rv5', 'mkm09_rv3', and 'mathis90_rv3.1'
        picklepath: Where your dereddened observations pickle will be saved.
        flux: BOOLEAN -- if True (1), the function will treat your photometry as flux units (erg s-1 cm-2).
              if False (0), the function will treat your photometry as being Flambda (erg s-1 cm-2 cm-1).
        lpath: Where the 'ext_laws.pkl' file is located. I suggest hard coding it as 'edgepath'.
        err_prop: BOOLEAN -- if True (1), will propagate the uncertainty of your photometry with the
                  uncertainty in your Av. Otherwise, it will not.
        UV: Uses dereddening law from Whittet et al. 2004 based on the extinction towards HD 29647
            for wavelengths between 0.125-9.33 microns.
            NOTE: This is ONLY useful for stars extincted by diffuse media, with RV = 3.1 (MATHIS LAW)
        clob: boolean, if set to True, will clobber the old pickle

        OUTPUT
        There will be a pickle file called '[self.name]_obs.pkl' in the path provided in picklepath. If
        there is already an obs pickle file there, it will add an integer to the name to differentiate
        between the two files, rather than overwriting.
        """

        # Read in the dereddening laws pickle. The default is whereever you keep EDGE.py, but you can move it.
        extPickle = open(lpath + 'ext_laws.pkl', 'rb')
        extLaws   = cPickle.load(extPickle, encoding = 'latin1')
        extPickle.close()

        # Figure out which law we will be using based on the user input and Av:
        if law == 'mkm09_rv5':
            print('Using the McClure (2009) ext. laws for Av >3\nwith the Mathis (1990) Rv=5.0 \
                   law for Av < 3\n(appropriate for molecular clouds like Ophiuchus).')
            AjoAks = 2.5341
            AvoAj  = 3.06

            if Av >= 8.0:                                       # high Av range
                wave_law = extLaws['mkm_high']['wl']
                ext_law  = extLaws['mkm_high']['ext'] / AjoAks
            elif Av >= 3.0 and Av < 8.0:                        # med Av range
                wave_law = extLaws['mkm_med']['wl']
                ext_law  = extLaws['mkm_med']['ext'] / AjoAks
            elif Av < 3.0:                                      # low Av range
                wave_law = extLaws['mathis_rv5']['wl']
                ext_law  = extLaws['mathis_rv5']['ext']
            else:
                raise ValueError('DERED: Specified Av is not within acceptable ranges.')
        elif law == 'mkm09_rv3':
            print('Using the McClure (2009) ext. laws for Av >3\nwith the Mathis (1990) Rv=3.1 \
                   law for Av < 3\n(appropriate for molecular clouds like Taurus).')
            AjoAks = 2.5341
            AvoAj  = 3.55 # for Rv=3.1 **WARNING** this is still the Rv=5 curve until 2nd step below.

            if Av >= 8.0:                                       # high Av range
                wave_law = extLaws['mkm_high']['wl']
                ext_law  = extLaws['mkm_high']['ext'] / AjoAks
            elif Av >= 3.0 and Av < 8.0:                        # med Av range
                wave_law = extLaws['mkm_med']['wl']
                ext_law  = extLaws['mkm_med']['ext'] / AjoAks
            elif Av < 3.0:                                      # low Av range
                wave_law = extLaws['mathis_rv3']['wl']
                ext_law  = extLaws['mathis_rv3']['ext']
            else:
                raise ValueError('DERED: Specified Av is not within acceptable ranges.')

            # Fix to the wave and ext. law:
            wave_jm  = extLaws['mathis_rv3']['wl']
            ext_jm   = extLaws['mathis_rv3']['ext']
            jindmkm  = np.where(wave_law >= 1.25)[0]
            jindjm   = np.where(wave_jm < 1.25)[0]
            wave_law = np.append(wave_jm[jindjm], wave_law[jindmkm])
            ext_law  = np.append(ext_jm[jindjm], ext_law[jindmkm])

        elif law == 'mathis90_rv3.1':
            print('Using the Mathis (1990) Rv=3.1 law\n(appropriate for diffuse ISM).')
            AjoAks   = 2.5341
            AvoAj    = 3.55                                     # for Rv=3.1
            wave_law = extLaws['mathis_rv3']['wl']
            ext_law  = extLaws['mathis_rv3']['ext']
        else:
            raise ValueError('DERED: Specified extinction law string is not recognized.')

        A_object        = Av
        A_object_string = str(round(A_object,2))

        # Open up a new TTS_Obs object to take the dereddened values:
        deredObs        = TTS_Obs(self.name)

        # Loop over the provided spectra (and possible errors), and compute the dereddened fluxes
        # and uncertainties where available. The possible uncertainties calculations are both the
        # SSC/SMART's "spectral uncertainty" and the "nod-differenced uncertainty".

        for specKey in self.spectra.keys():
            extInterpolated = np.interp(self.spectra[specKey]['wl'], wave_law, ext_law) # Interpolated ext.

            #If the UV flag is on, replace the extinction law between 0.125 - .33 microns with a different law
            if UV == True:

                #Ensure that you are using the Rv = 3.1 Mathis law
                if law != 'mathis90_rv3.1' and law != 'mkm09_rv3':
                    raise ValueError('UV dereddening mode for use only with the low extinction laws (mathis90_rv3.1 and mkm09_rv3)')

                #Covert wavelength to 1/microns
                x = self.spectra[specKey]['wl'] ** (-1)

                #Define the valid range (3-8 (micron)^-1)
                UVrange = np.where((x > 3) & (x < 8))

                # If the range does contain some points in the right wavelength range, calculate the new extinction law there
                if len(UVrange[0]) != 0:

                    #Because this function only works for diffuse regions, forced to use an Rv of 3.1 (common interstellar value)
                    Rv = 3.1

                    # Define the extinction parameters (HD 29647, Table 1, Whittet et al. 2004)
                    # Functional form of the extinction from Fitzpatrick + Massa (1988, 1990)
                    c1 = 0.005
                    c2 = 0.813
                    c3 = 3.841
                    c4 = 0.717
                    x0 = 4.650
                    gamma = 1.578

                    #Calculate the extinction at each wavelength in terms of Alambda/Av
                    #Uses parametrization set up by Fitzpatrick + Massa (1988, 1990)

                    #Drude function at x0 with width gamma
                    D = (x**2) / ((x**2 - x0**2)**2 + gamma**2*x**2)
                    #Polynomial representing far UV rise
                    F = 0.5392*(x-5.9)**2 + 0.0564*(x-5.9)**3
                    F[np.where(x<5.9)] = 0

                    #Calculate A_UV and truncate outside the valid range (3-8 (micron)^-1)
                    ext_UV_all = 1 + (c1 + c2*x + c3*D + c4*F)/Rv

                    ext_UV = ext_UV_all[UVrange[0]]

                    #Convert from Alam/Av to Alam/Aj to be consitant with extInterpolated
                    ext_UV_j = ext_UV * AvoAj

                    #Replace the old values of extinction with the new ones
                    extInterpolated[UVrange[0]] = ext_UV_j

                else:
                    pass


            A_lambda        = extInterpolated * (A_object / AvoAj)
            spec_flux       = np.float64(self.spectra[specKey]['lFl']*10.0**(0.4*A_lambda))



            if 'err' in self.spectra[specKey].keys():
                spec_unc    = np.float64(spec_flux*np.sqrt(((self.spectra[specKey]['err']/self.spectra[specKey]['lFl'])\
                                         **2.) + (((0.4*math.log(10)*extInterpolated*Av_unc)/(AvoAj))**2.)) )
            else:
                spec_unc    = None

            # Correct units to flux:
            #Apparently I don't need this anymore?
#            spec_flux       = spec_flux * self.spectra[specKey]['wl'] * 1e-4

#            if spec_unc != None:
#                spec_unc    = spec_unc  * self.spectra[specKey]['wl'] * 1e-4

            deredObs.add_spectra(specKey, self.spectra[specKey]['wl'], spec_flux, errors=spec_unc)


        # Spectra are done, onwards to photometry:
        for photKey in self.photometry.keys():
            extInterpolated = np.interp(self.photometry[photKey]['wl'], wave_law, ext_law)

            #If the UV flag is on, replace the extinction law between 0.125 - .33 microns with a different law
            if UV == True:

                #Ensure that you are using the Rv = 3.1 Mathis law
                if law != 'mathis90_rv3.1' and law != 'mkm09_rv3':
                    raise ValueError('UV dereddening mode for use only with the low extinction laws (mathis90_rv3.1 and mkm09_rv3)')

                #Covert wavelength to 1/microns
                x = self.photometry[photKey]['wl'] ** (-1)

                #Define the valid range (3-8 (micron)^-1)
                UVrange = np.where((x > 3) & (x < 8))

                # If the range does contain some points in the right wavelength range, calculate the new extinction law there
                if len(UVrange[0]) != 0:

                    #Because this function only works for diffuse regions, forced to use an Rv of 3.1 (common interstellar value)
                    Rv = 3.1

                    # Define the extinction parameters (HD 29647, Table 1, Whittet et al. 2004)
                    # Functional form of the extinction from Fitzpatrick + Massa (1988, 1990)
                    c1 = 0.005
                    c2 = 0.813
                    c3 = 3.841
                    c4 = 0.717
                    x0 = 4.650
                    gamma = 1.578

                    #Calculate the extinction at each wavelength in terms of Alambda/Av
                    #Uses parametrization set up by Fitzpatrick + Massa (1988, 1990)

                    #Drude function at x0 with width gamma
                    D = (x**2) / ((x**2 - x0**2)**2 + gamma**2*x**2)
                    #Polynomial representing far UV rise
                    F = 0.5392*(x-5.9)**2 + 0.0564*(x-5.9)**3
                    F[np.where(x<5.9)] = 0

                    #Calculate A_UV and truncate outside the valid range (3-8 (micron)^-1)
                    ext_UV_all = 1 + (c1 + c2*x + c3*D + c4*F)/Rv

                    ext_UV = ext_UV_all[UVrange[0]]

                    #Convert from Alam/Av to Alam/Aj to be consitant with extInterpolated
                    ext_UV_j = ext_UV * AvoAj

                    #Replace the old values of extinction with the new ones
                    extInterpolated[UVrange[0]] = ext_UV_j


                else:
                    pass

            A_lambda        = extInterpolated * (A_object / AvoAj)
            if flux:
                photcorr    = self.photometry[photKey]['lFl'] / (self.photometry[photKey]['wl']*1e-4)
            else:
                photcorr    = self.photometry[photKey]['lFl']
            phot_dered      = np.float64(photcorr*10.0**(0.4*A_lambda))
            if 'err' in self.photometry[photKey].keys():
                if flux:
                    errcorr = self.photometry[photKey]['err'] / (self.photometry[photKey]['wl']*1e-4)
                else:
                    errcorr = self.photometry[photKey]['err']
                if err_prop:
                    phot_err= np.float64(photcorr * np.sqrt(((errcorr/photcorr)**2.) + \
                                         (((0.4*math.log(10.)*extInterpolated*Av_unc)/AvoAj)**2.)) )
                else:
                    phot_err= np.float64(errcorr*10.0**(0.4*A_lambda)) # Without propogating error!
            else:
                phot_err    = None
            if photKey in self.ulim:
                ulimVal     = 1
            else:
                ulimVal     = 0
            # Now, convert everything to flux units:
            phot_dered      = phot_dered * self.photometry[photKey]['wl'] * 1e-4
            try:
                phot_err        = phot_err * self.photometry[photKey]['wl'] * 1e-4
            except TypeError:
                pass
            deredObs.add_photometry(photKey, self.photometry[photKey]['wl'], phot_dered, errors=phot_err, ulim=ulimVal, verbose = 0)



        # Now that the new TTS_Obs object has been created and filled in, we must save it:
        deredObs.SPPickle(picklepath=picklepath, clob = clob)

        return


    def SPPickle(self, picklepath, clob = False, fill = 3):
        """
        The new version of SPPickle, different so you can differentiate between red and dered pickles.

        INPUT
        picklepath: The path where the new pickle file will be located.
        clob: boolean, if set to True, will clobber the old pickle
        fill: How many numbers used in the model files (4 = name_XXXX.fits).

        OUTPUT
        A new pickle file in picklepath, of the name [self.name]_red.pkl
        """

        # Check whether or not the pickle already exists:
        pathlist        = filelist(picklepath)
        outname         = self.name + '_red.pkl'
        count           = 1
        while 1:
            if outname in pathlist and clob == False:
                if count == 1:
                    print('SPPICKLE: Pickle already exists in directory. For safety, will change name.')
                countstr= str(count).zfill(fill)
                count   = count + 1
                outname = self.name + '_red_' + countstr + '.pkl'
            else:
                break
        # Now that that's settled, let's save the pickle.
        f               = open(picklepath + outname, 'wb')
        cPickle.dump(self, f)
        f.close()
        return
