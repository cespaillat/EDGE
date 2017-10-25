#!/usr/bin/env python

from astropy.io import ascii
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.interpolate as sinterp
import os

utilpath        = os.path.dirname(os.path.realpath(__file__))+'/'
commonpath     = utilpath+'/../COMMON/'

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

def interp(x0, x, y, n,i0):
	"""
	Does a linear interpolation at x0 on arrays x and y.
	
	INPUTS
		x0: value you want to interpolate at
		x:  x array
		y:  y array
		n:  number of items in array
		i0: label of first value
	
	OUTPUT
		interp: interpolated value from y array

	EXAMPLE
		flux_phot=util.interp(wl_phot[i],xwef,save_xfluxb,12,i0)
		For a given wavelength (wl_phot[i]) locate and interpolate 
		in an array of wavelengths (xwef), look up and interpolate 
		for the corresponding flux (save_xfluxb) and return this
		value

	"""

	if x[1] < x[0] :
	# x in decreasing order
		while x0 < x[i0]:
			i0 = i0 + 1
	else:
	# x in increasing order
		while x0 > x[i0]:
			i0 = i0 + 1

	if i0 > 0 : 
		i0 = i0 - 1
		
	interp = (((y[i0] * (x[i0 + 1] - x0)) \
	+ (y[i0 + 1] * (x0 - x[i0]))) / (x[i0 + 1] - x[i0]))
     
	return interp

def temp_structure(model, figure_path=None, bw=False):
    """ Produces a figure of the temperature structure of the disk.
    
    Purpose:
    --------
    This function takes a collated .fits model and creates a pdf figure of its
    temperature structure.

    Parameters:
    -----------
    model : string
        Path of the .fits collated model.
    figure_path : string.
        Path (and optionally, name) of the figure. Default is None.
        Options for this parameter include (do not forget the .pdf extension when necessary):
        - If no figure_path is specified, the figure will be saved in the current directory
        with the original model name.
        - If only a name is specified (e.g. figure.pdf), the figure will be in the current
        directory with the provided filename name.
        - If a path with no figure name (i.e. /path/to/figure/) is provided, the figure will
        be saved in that path with the original model name. Do not forget the slash at the end!
        - If the whole path and figure name are specified (i.e. /path/to/figure/figure.pdf),
        the figure will be saved there with the corresponding name.
    bw : bool
        If True, the figure lines will be black dotted and dashed lines for clarity when
        printed in bacl and white. Default is false.

    Example:
    --------
    The temp_structure function takes a collated DIAD model to produce a figure of three temperatures 
    (midplane, tau=2/3, and surface) at different radii. It takes three parameters: the path to the 
    collated diad model (model), the path and/or name of the figure (figure_path, optional), and 
    whether the figure should be in black and white or not (bw, optional, default is False). You may 
    simply specify one collated model (e.g. model.fits) and run:

    my_model = 'model_001.fits'
    temp_structure(my_model)

    This will produce a 'model_001.pdf' figure in the current directory.

    You may specify the path, name, or both (combined) for the output figure, i.e.:

    my_model = 'model_001.fits'
    figure_path = 'figure1.pdf'
    figure_path = '/path/to/figures/'
    figure_path = '/path/to/figures/figure1.pdf'
    # any of these will work
    temp_structure(my_model, figure_path=figure_path)

    are all valid inputs. In the first case, you will get a 'figure1.pdf' figure in the current directory. 
    In the second case, you will get a figure called 'model_001.pdf' in '/path/to/figures/'. And in the 
    third case, 'figure1.pdf' will be created in '/path/to/figures/'.

    Also, you may choose to produce the figure in black and white for better clarity when printed. 
    In that case, set the 'bw' parameter to True, i.e.:

    my_model = 'model_002.fits'
    figure_path = '/path/to/figures/figure2.pdf'
    temp_structure(my_model, figure_path=figure_path, bw=True)

    Will produce a figure in black and white at '/path/to/figures/figure2.pdf'.
    
    Description of temperatures stored in collated models:
    ------------------------------------------------------
        The temperatures are stored in the second layer of the fits files. Each index contains:
        0, RADIUS: radii (AUs)
        1, TEFF: temperature related to disk viscosity. Not very useful.
        2, TIRR: temperature related to the irradiation on the disk. Not very useful.
        3, TZEQ0: temperature at z=0, which is the disk midplane.
        4, TZEQZMX: temperature at z=infinity, which is the upper layer of the disk.
        5, TZEQZS: temperature at the irradiation surface, which corresponds to tau = 2/3
        6, TZTAU: temperature at surface defined by tau = 1.
    """

    # create the figure name if needed
    # 1) nothing specified
    if figure_path is None:
        figname = os.path.basename(model+'.pdf').replace('.fits', '')
    else:
        basename = os.path.basename(figure_path)
        # 2) look for slashes to know if a path has been specified or just a name
        if '/' in figure_path:
            # 3) if there is a basename after the path, then the user provided a
            # a full path
            if len(basename) > 0:
                figname = figure_path
            # 4) if not, only the path was specified
            else:
                figname = figure_path + os.path.basename(model+'.pdf')
        # 5) in this case, only a figure name was specified
        else:
            figname = figure_path

    # read model and extract temperatures
    model = fits.open(model)
    radii = model[1].data[0]
    t_midplane = model[1].data[3]
    t_surface = model[1].data[4]
    t_tau_twothirds = model[1].data[5]

    # make figure (no interactive plotting needed)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    # plot the profiles
    # this is the color case
    if bw is False:
        plt.plot(radii, t_midplane, label=r'$T_{\rm midplane}$')
        plt.plot(radii, t_surface, label=r'$T_{z \infty}$')
        plt.plot(radii, t_tau_twothirds, label=r'$T_{\tau=2/3}$')
    # black and white case
    else:
        plt.plot(radii, t_midplane, 'k:', label=r'$T_{\rm midplane}$')
        plt.plot(radii, t_surface, 'k-', label=r'$T_{z \infty}$')
        plt.plot(radii, t_tau_twothirds, 'k--', label=r'$T_{\tau=2/3}$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Radius (AU)', size=16)
    plt.ylabel('T (K)', size=16)
    plt.legend(loc='best', fontsize=14)
    ax.axes.tick_params(axis='both', labelsize=13)
    # save figure and close it
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close()
    
    return

#-------------------------------------------------------------
def parameters_siess(tstar,lui,commonpath=commonpath):
	"""
	Calculates age and mass using Siess & Forestini 1996 tracks
	
	INPUTS
		tstar: stellar temperature
		lui:  stellar luminosity
	
	OUTPUT
		mass: stellar mass
		age: stellar age

	"""
	
	nages=6
	nmasses=29
	lsun=4e33
	rsun=7e10
	i0=0
	
	lstar=np.log10(lui)
	tlstar=np.log10(tstar)

	# reads in tracks and corresponding ages	
	fileiso=(commonpath+'isochrones/'+'isochron3e5', \
	commonpath+'isochrones/'+'isochron1e6', \
	commonpath+'isochrones/'+'isochron3e6', \
	commonpath+'isochrones/'+'isochron1e7', \
	commonpath+'isochrones/'+'isochron3e7', \
	commonpath+'isochrones/'+'isochron1e8')	
	ages=3.e5,1.e6,3.e6,1.e7,3.e7,1.e8

	# for the input tstar this interpolates between the T and L columns to get L 
	# for each isochrone file and stores these 6 values in lu_track_temp 
	lu_track_temp = []
	for i in range(0,nages):
		tablesiess=ascii.read(fileiso[i], header_start=2, data_start=3, data_end=32)  
		lint=np.log10(tablesiess['L'])
		tint=np.log10(tablesiess['Teff'])
		massint=tablesiess['Mass']
		lu_track_temp.append(interp(tlstar,tint,lint,nmasses,i0))

	# if out of isochrone range, reject		
	if lstar > lu_track_temp[0] or lstar < lu_track_temp[5]:
		print('Luminosity out of isochrone bounds')
		mass=99.0
		age=99.0e6
		return mass,age
		
	# to avoid issues due to extrapolation of tracks to high L
	for jj in range (0,nages):
		if lu_track_temp[jj] > lstar:
			i0=jj

	# this compares the input L (lstar) to the L from each isochrone file 
	# for the appropriate Teff (lu_track_temp) and iterpolates to find the age
	age=interp(lstar,lu_track_temp,ages,nages,i0)

	# for each of the 6 isochrone files, this grabs the L for each M
	# then it interpolates using the calculated age to get the L for each M
	# it does this for each of the 29 masses listed in each isochrone file
	# and stores in lu_track_mass
	lint= []
	lu_track_mass = []
	for i2 in range(0,nmasses):
		for i in range(0,nages):
			tablesiess=ascii.read(fileiso[i], header_start=2, data_start=i2+3, data_end=i2+4)  
			lint.append(np.log10(tablesiess['L']))
			i0=0
		lu_track_mass=np.append(lu_track_mass,interp(age,ages,lint,nages,i0))
		lint= []
		
	# this compares the input L to the track L and M interpolated at the calculated age
	# (i.e., lu_track_mass) and interpolates to get the M
	mass=interp(lstar,lu_track_mass,massint,nmasses,i0)
	
	return mass,age

#-------------------------------------------------------------
def parameters_baraffe(tstar,lui,commonpath=commonpath):
	"""
	Calculates age and mass using Baraffe et al. 1998 tracks
	
	INPUTS
		tstar: stellar temperature
		lui:  stellar luminosity
	
	OUTPUT
		mass: stellar mass
		age: stellar age

	"""
	
	nages=19
	nmasses=37
	lsun=4e33
	rsun=7e10
	i0=0

	lstar=np.log10(lui)
	tlstar=np.log10(tstar)

	lages=[6.0,6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9, \
	7.0,7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8]

	ages=[10**6.0,10**6.1,10**6.2,10**6.3,10**6.4, \
	10**6.5,10**6.6,10**6.7,10**6.8,10**6.9, \
	10**7.0,10**7.1,10**7.2,10**7.3,10**7.4, \
	10**7.5,10**7.6,10**7.7,10**7.8]

	# for the input tstar this interpolates between the T and L columns to get L 
	# for each isochrone and stores these 19 values in lu_track_temp 
	#BCAH98_iso.1 has Teff=2542-3978, M 0.02-1.2, l/H=1
	#BCAH98_iso.3 has Teff=3862-4521, M 0.7-1.2, l/H=1.9
	#here we use file BCAH98_iso.3 with l/H=1.9 for 0.7 msun and above
	lu_track_temp = []
	for i in range(0,nages):
		if i==0:
			start1=19	
			tablebaraffe1=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.1', data_start=start1, data_end=start1+28)	
			lint1=tablebaraffe1['col4'] #already in log10
			tint1=np.log10(tablebaraffe1['col2'])
			massint1=tablebaraffe1['col1']
			start3=19	
			tablebaraffe3=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.3', data_start=start3, data_end=start3+11)	
			lint3=tablebaraffe3['col4'] #already in log10
			tint3=np.log10(tablebaraffe3['col2'])
			massint3=tablebaraffe3['col1']
			lint=np.concatenate((lint1,lint3),axis=0)
			tint=np.concatenate((tint1,tint3),axis=0)
			massint=np.concatenate((massint1,massint3),axis=0)
		else:
			start1=start1+nmasses+5
			tablebaraffe1=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.1', data_start=start1, data_end=start1+28)	
			lint1=tablebaraffe1['col4'] #already in log10
			tint1=np.log10(tablebaraffe1['col2'])
			massint1=tablebaraffe1['col1']
			start3=start3+11+5
			tablebaraffe3=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.3', data_start=start3, data_end=start3+11)	
			lint3=tablebaraffe3['col4'] #already in log10
			tint3=np.log10(tablebaraffe3['col2'])
			massint3=tablebaraffe3['col1']
			lint=np.concatenate((lint1,lint3),axis=0)
			tint=np.concatenate((tint1,tint3),axis=0)
			massint=np.concatenate((massint1,massint3),axis=0)
		lu_track_temp.append(interp(tlstar,tint,lint,nmasses,i0))

	# if out of isochrone range, reject		
	if lstar > lu_track_temp[0] or lstar < lu_track_temp[5]:
		print('Luminosity out of isochrone bounds')
		mass=99.0
		age=99.0e6
		return mass,age

	# to avoid issues due to extrapolation of tracks to high L
	for jj in range (0,nages):
		if lu_track_temp[jj] > lstar:
			i0=jj

	# this compares the input L (lstar) to the L from each isochrone file 
	# for the appropriate Teff (lu_track_temp) and iterpolates to find the age
	age=interp(lstar,lu_track_temp,ages,nages,i0) #in log10

	#THIS TAKES TOO LONG, WAY TO SPEED UP?
	#for each of the 19 isochrones, this grabs the L for each M
	# then it interpolates using the calculated age to get the L for each M
	# it does this for each of the 37 masses listed in each isochrone file
	# and stores in lu_track_mass
	lint= []
	lu_track_mass = []
	for i2 in range(0,nmasses):
		for i in range(0,nages):
			if i2<28:
				start1=19+i2+(i*(nmasses+5))
				tablebaraffe1=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.1', data_start=start1, data_end=start1+1)	
				lint.append(tablebaraffe1['col4']) #already in log10
			else:
				start1=19+(i2-28)+(i*(11+5))	
				tablebaraffe3=ascii.read(commonpath+'isochrones/'+'BCAH98_iso.3', data_start=start1, data_end=start1+1)	
				lint.append(tablebaraffe3['col4']) #already in log10
		lu_track_mass=np.append(lu_track_mass,interp(age,ages,lint,nages,i0))
		lint= []

	# this compares the input L to the track L and M interpolated at the calculated age
	# (i.e., lu_track_mass) and interpolates to get the M
	mass=interp(lstar,lu_track_mass,massint,nmasses,i0)
	
	return mass,age

#-------------------------------------------------------------
def redd(av,wl,commonpath=commonpath):
	"""
	Calculates reddening correction using Mathis ARAA 28,37,1990
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=39
	
	tablemathis=ascii.read(commonpath+'ext_laws/'+'mathis.table.rev', delimiter=" ")    	 
	wlm=tablemathis['wlmu']
	al=tablemathis['A_wl/A_J_R3.1']
     
	# normalize to V
	avv=al[wlm == 0.55]
	al=al/avv

	redd=interp(wl,wlm,al,nm,i0)*av

	return redd
	
#-------------------------------------------------------------
def reddhd(av,wl,commonpath=commonpath):
	"""
	Calculates reddening correction using HD29647
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=101
	
	table=ascii.read(commonpath+'ext_laws/'+'hd29647_ext_pei_1.dat', delimiter=" ")    	 
	wlm=table['col1']
	al=table['col2']
	wlm=wlm*1e-4
     
	redd=interp(wl,wlm,al,nm,i0)*av

	return redd
	
#-------------------------------------------------------------
def reddccm89(wl,r,commonpath=commonpath):
	"""
	Calculates reddening correction using Crdelli, Clayton, & Mathis 1989, ApJ, 345, 245
	
	INPUTS
		av: visual extinction
		r: extinction factor
	
	OUTPUT
		alambda_cc8: reddening correction

	"""

	x=1/wl

	if x<0.3:
		alambda_cc8=0.
		return alambda_cc8

	if x>10:
		print('outside CCM89 range')
		stop

	if x>0.3 and x<1.1:
		a=0.574*x**1.61
		b=-0.527*x**1.61

	else:
		if x>1.1 and x<3.3:
			y=x-1.82
			a=1.+y*(0.176999+y*(-0.50447+y*(-0.02427+y*(0.72085 \
			+ y*(0.01979+y*(-0.77530+y*0.32999))))))
			b=y*(1.41338+y*(2.28305+y*(1.07233+y*(-5.38434 \
			+ y*(-0.62251+y*(5.30260-y*2.09002))))))
		if x>3.3 and x<8:
			if x>5.9 and x<8:
				fa=-0.0447*(x-5.9)**2-0.009779*(x-5.9)**3
				fb=0.2130*(x-5.9)**2+0.1207*(x-5.9)**3
			else:
				fa=0.
				fb=0.
			a=1.752-0.316*x-0.104/((x-4.67)**2+0.341)+fa
			b=-3.090+1.825*x+1.206/((x-4.62)**2+0.263)+fb

		if x>8 and x<10:
			a=-1.073+(x-8.)*(-0.628+(x-8.)*(0.137-(x-8.)*0.070))
			b=13.670+(x-8.)*(4.257+(x-8.)*(-0.420+(x-8.)*0.374))
	
	alambda_cc8=a+b/r

	return alambda_cc8
	
#-------------------------------------------------------------
def reddmcclureavgt8rv5p0(av,wl,commonpath=commonpath):
	"""
	Calculates reddening correction using McClure (2010) 
	for Av>8 & Rv=5
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=22
	
	table=ascii.read(commonpath+'ext_laws/'+'mcclurereddening_avgt8_rv5p0', data_start=1)    	 
	wlm=table['col1']
	al=table['col2']
     
	redd=interp(wl,wlm,al,nm,i0)*av

	return redd
	
#-------------------------------------------------------------
def reddmcclureavgt8rv3p1(av,wl,commonpath=commonpath):
	"""
	Calculates reddening correction using McClure (2010)
	for Av>8 & Rv=3.1
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=22
	
	table=ascii.read(commonpath+'ext_laws/'+'mcclurereddening_avgt8_rv3p1', data_start=1)    	 
	wlm=table['col1']
	al=table['col2']
     
	redd=interp(wl,wlm,al,nm,i0)*av

	return redd
	
#-------------------------------------------------------------
def reddmcclureavlt8rv5p0(av,wl,commonpath):
	"""
	Calculates reddening correction using McClure (2010)
	for Av<8 & Rv=5
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=22
	
	table=ascii.read(commonpath+'ext_laws/'+'mcclurereddening_avlt8_rv5p0', data_start=1)    	 
	wlm=table['col1']
	al=table['col2']
     
	redd=interp(wl,wlm,al,nm,i0)*av

	return redd
	
#-------------------------------------------------------------
def reddmcclureavlt8rv3p1(av,wl,commonpath):
	"""
	Calculates reddening correction using McClure (2010)
	for Av<8 & Rv=3.1
	
	INPUTS
		av: visual extinction
		wl: wavelength
	
	OUTPUT
		redd: reddening correction

	"""

	i0=0
	nm=22
	
	table=ascii.read(commonpath+'ext_laws/'+'mcclurereddening_avlt8_rv3p1', data_start=1)    	 
	wlm=table['col1']
	al=table['col2']
     
	redd=interp(wl,wlm,al,nm,i0)*av

	return redd




