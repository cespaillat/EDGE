#!/usr/bin/env python

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as sinterp
import os

utilpath = os.path.dirname(os.path.realpath(__file__))+'/'
commonpath = utilpath+'../COMMON/'

#-------------------------------------------------------------

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

def convertJy_to_Mag(value, band, getwl='False'):
    """
    This function was adapted from the above convertMag function by A. Rilinger
    Converts a flux in Jy into a magnitude. To use this for an array, use np.vectorize().
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
    value: A flux value in Jy.
    band: The band corresponding to the flux value.
    getwl: Boolean -- If True, output wavelength value in addition to the
            magnitude.  If False, only magnitude is returned.

    OUTPUTS
    fluxmag: The flux value in magnitudes.
    """

    # Convert to magnitudes:
    if band.upper()     == 'U':
        fluxmag       = -2.5 * np.log10(value / 1810.)
        wavelength  = 0.367                                     # In Microns
    elif band.upper()   == 'B':
        fluxmag       = -2.5 * np.log10(value / 4260.)
        wavelength  = 0.436
    elif band.upper()   == 'V':
        fluxmag       = -2.5 * np.log10(value / 3640.)
        wavelength  = 0.545
    elif band.upper()   == 'R':
        fluxmag       = -2.5 * np.log10(value / 3080.)
        wavelength  = 0.638
    elif band.upper()   == 'I':
        fluxmag       = -2.5 * np.log10(value / 2550.)
        wavelength  = 0.797
    elif band.upper()   == 'J':
        fluxmag       = -2.5 * np.log10(value / 1600.)
        wavelength  = 1.220
    elif band.upper()   == 'H':
        fluxmag       = -2.5 * np.log10(value / 1080.)
        wavelength  = 1.630
    elif band.upper()   == 'K':
        fluxmag       = -2.5 * np.log10(value / 670.)
        wavelength  = 2.190
    elif band.upper()   == 'L':
        fluxmag       = -2.5 * np.log10(value / 281.)
        wavelength  = 3.450
    elif band.upper()   == 'M':
        fluxmag       = -2.5 * np.log10(value / 154.)
        wavelength  = 4.750
    elif band.upper()   == 'N':
        fluxmag       = -2.5 * np.log10(value / 37.)
        wavelength  = 10.10
    elif band.upper()   == 'Q':
        fluxmag       = -2.5 * np.log10(value / 10.)
        wavelength  = 20.00
    elif band.upper()   == 'SDSSG':
        fluxmag       = -2.5 * np.log10(value / 3730.)
        wavelength  = 0.4686
    elif band.upper()   == 'SDSSR':
        fluxmag       = -2.5 * np.log10(value / 4490.)
        wavelength  = 0.6165
    elif band.upper()   == 'SDSSI':
        fluxmag       = -2.5 * np.log10(value / 4760.)
        wavelength  = 0.7481
    elif band.upper()   == 'SDSSZ':
        fluxmag       = -2.5 * np.log10(value / 4810.)
        wavelength  = 0.8931
    elif band.upper()   == 'MIPS24':
        fluxmag       = -2.5 * np.log10(value / 7.17)
        wavelength  = 23.675
    elif band.upper()   == 'MIPS70':
        fluxmag       = -2.5 * np.log10(value / 0.778)
        wavelength  = 71.42
    elif band.upper()   == 'MIPS160':
        fluxmag       = -2.5 * np.log10(value / 0.16)
        wavelength  = 155.9
    elif band.upper()   == 'IRAC3.6':
        fluxmag       = -2.5 * np.log10(value / 280.9)
        wavelength  = 3.60
    elif band.upper()   == 'IRAC4.5':
        fluxmag       = -2.5 * np.log10(value / 179.7)
        wavelength  = 4.50
    elif band.upper()   == 'IRAC5.8':
        fluxmag       = -2.5 * np.log10(value / 115.)
        wavelength  = 5.80
    elif band.upper()   == 'IRAC8.0':
        fluxmag       = -2.5 * np.log10(value / 64.13)
        wavelength  = 8.0
    elif band.upper()   == 'W1':
        fluxmag       = -2.5 * np.log10(value / 309.5)
        wavelength  = 3.35
    elif band.upper()   == 'W2':
        fluxmag       = -2.5 * np.log10(value / 171.8)
        wavelength  = 4.60
    elif band.upper()   == 'W3':
        fluxmag       = -2.5 * np.log10(value / 31.67)
        wavelength  = 11.56
    elif band.upper()   == 'W4':
        fluxmag       = -2.5 * np.log10(value / 8.36)
        wavelength  = 22.09
    elif band.upper() == 'GAIAG':
        fluxmag       = -2.5 * np.log10(value / 3488.)
        wavelength = .550


    else:
        raise ValueError('CONVERTMAG: Unknown Band given. Cannot convert.')


    if getwl == True:
        return fluxmag, wavelength
    else:
        return fluxmag

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

    # Calculate the U flux for the photosphere and star to get excess luminosity:
    photFlux= convertMag(Uphot, 'U') * 1e-3 * (0.068 / 0.367)
    starFlux= convertMag(Umag, 'U') * 1e-3 * (0.068 / 0.367)

    L_u     = (4.0*math.pi) * (starFlux - photFlux) * (d_pc * 3.086e16)**2.0
    L_u_norm= L_u / 3.84e26

    # Use the U band luminosity to calculate the accretion luminosity:
    L_acc   = 3.84e26 * 10.0**(1.09*math.log10(L_u_norm) + 0.98)

    # Lastly, back out the accretion rate:
    G       = 6.67e-11                              # G in meters
    #Rin_m   = Rin * 1.496e11                        # Rin in meters
    Rstar_m = Rstar * 6.955e8                       # Rstar in meters
    Mstar_kg= Mstar * 1.989e30                      # Mstar in kg
    Mdot    = (Rstar_m * L_acc / (G*Mstar_kg)) / 0.8 * 3.16e7 / 1.989e30

    return Mdot

def significant_digit(value):
    '''
    Returns the significant digit (order of mag) of a number
    '''
    return int(np.floor(np.log10(abs(value))))


def round_to_significant(value, uncertainty):
    '''
    PURPOSE:
        Returns a value and its uncertainty rounded to the corresponding significant

    INPUTS:
        value:[float] Value of the measurement
        uncertainty:[float] Uncertainty in the measurement.

    NOTES:
        If the leading sigficant digit is 1 or 2, then keep the proceeding significant digits
        If the leading significant digit is above 3 or above, then round to the place of the leading significant digit

    AUTHOR:
        Sierra Grant, October 10th, 2017
    '''

    if (np.isfinite(value) * np.isfinite(uncertainty)) == 0 or value == 0 or uncertainty  == 0:
        return 'nan', 'nan'

    value, uncertainty = np.float(value), np.float(uncertainty)
    sd = significant_digit(uncertainty)

    #Handle the weird case with numbers smaller than 10.
    if sd == 0:
        if uncertainty > 0:
            sd_val = float(str(uncertainty)[0])
        if uncertainty < 0:
            sd_val = float(str(uncertainty)[1])
    else:
        sd_str = ('{:1.'+str(int(np.abs(sd)+5))+'f}').format(uncertainty)
        sd_val = float(sd_str[sd_str.index(".")-sd])

    #Handle numbers that round to 1 digit
    if sd_val>2.0:
        #print(uncertainty,sd,sd_val,'I AM GREATER THAN 2')
        # round value and uncertainty (we used -1 in front of sd to change the behaviour of round)
        outval = np.round(value, -1 * sd)
        outunc = np.round(uncertainty, -1 * sd)

        #Handle errors greater than 1
        if sd >= 0:
            return ("{:.0f}").format(outval),("{:.0f}").format(outunc)
        else:
            return ("{:."+str(-1*sd)+"f}").format(outval),("{:."+str(-1*sd)+"f}").format(outunc)

    #Handle numbers that round to 2 digits
    if sd_val<=2.0:
        outval = np.round(value, -1 * sd + 1)
        outunc = np.round(uncertainty, -1 * sd + 1)

        #Handle errors greater than 10
        if sd >=1:
            return ("{:.0f}").format(outval),("{:.0f}").format(outunc)

        #Handles the one weird case with rounding to 3.0 exactly
        if outunc == 3.0:
            return ("{:."+str(-1*sd)+"f}").format(outval),("{:."+str(-1*sd)+"f}").format(outunc)

        #Fix the issue where 3 is rounded
        if ('{:1.'+str(int(np.abs(sd)+5))+'f}').format(outunc)[('{:1.'+str(int(np.abs(sd)+5))+'f}').format(outunc).index(".")-sd] == '3':
            return ("{:."+str(-1*sd)+"f}").format(outval),("{:."+str(-1*sd)+"f}").format(outunc)
        else:
            return ("{:."+str(-1*sd+1)+"f}").format(outval),("{:."+str(-1*sd+1)+"f}").format(outunc)

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
