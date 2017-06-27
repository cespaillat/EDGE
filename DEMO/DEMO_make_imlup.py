import numpy as np
import EDGE as edge
from astropy.io import fits
from astropy.io import ascii
from astropy.io import votable

'''
DEMO_make_imlup.py
    
PURPOSE:
    Script that loads data of IM Lup into a EDGE pickle file
    
OUTPUT:
    A de-reddened pickle, ready for use with DEMO_analysis_imlup.py
    
NOTES: 
    If you are trying to re-run this code, you will need to change the paths
    
    This code puts all the photometry under a single group ('Literature') for ease of use.
    HOWEVER, This does not use the full power of EDGE, which allows you to have different names for each set of observations.
    This will be an issue if you are trying to use filter deconvolution when computing chi^2.
    Generally, groups of observations are separated by instrument (e.g., 'HST', 'MIPS', 'WISE')
    This can be done by keeping your observations separating and setting an instrument in the 'red.add_photometry' command
    
    The warning messages from the VOT table are normal.
    
    
AUTHOR:
    Connor Robinson, June 19th, 2017
'''

#Set up paths IF YOU ARE RE-RUNNING, YOU WILL NEED TO CHANGE THESE
specpath   = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/data/'
photpath   = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/data/'
picklepath = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/data/'

#Build a Red_Obs object to store the data
obj = 'imlup'
red = edge.Red_Obs(obj)

#******************************
#IRS Spectrum
#******************************
#Load in the IRS spectrum
irs = fits.open(specpath+'cassis_yaaar_opt_27064320_1.fits')

#Grab the wavelength
irswl = irs[0].data[:,0]

#Convert fluxes in ergs/s/cm^2
irs_flux = edge.convertJy(irs[0].data[:,1], irs[0].data[:,0])
irs_ferr = edge.convertJy(irs[0].data[:,3], irs[0].data[:,0])
irs_nod  = edge.convertJy(irs[0].data[:,4], irs[0].data[:,0])

#Add the errors in quadrature (only for IRS)
irs_err = np.sqrt(irs_ferr**2 + irs_nod**2)

#Sort array by wavelength.
sort = np.argsort(irs[0].data[:,0])

#Add the data
red.add_spectra('IRS', irs[0].data[:,0][sort], irs_flux[sort])#, errors = irs_err)

#******************************
#Photometry
#******************************
c = 3e14 #microns/s

#The photometry here is stored in a VOT table, but in practice any data format will work if you turn it into a wavelength
#array, a flux array and an error array.

#Data is mostly from Vizier, removed IRAS:100 points, and several HIP:Hp points which all looked incorrect
vo_obj = votable.parse(photpath+'imlup_viz.vot') 
data = vo_obj.get_first_table().array.data

#Get the fluxes + wavelength information and convert to microns
vizfilt = data['sed_filter']
vizphotjy = data['sed_flux']
vizwl   = c/(data['sed_freq']*1e9)
vizphoterrjy = data['sed_eflux']

#Convert photometryfluxes  to ergs/s/cm^2
vizphot = edge.convertJy(vizphotjy, vizwl)
vizphoterr = edge.convertJy(vizphoterrjy, vizwl)

#Add some data from ALMA manually, and convert into the correct units
almawl = np.array([890, (3e8/258e9)*1e6]) #um
almaflux = edge.convertJy(np.array([600.0 * 1e-3, 276*1e-3]), almawl)
almaerr = edge.convertJy(np.array([90.0 *1e-3, 2*1e-3]), almawl)

#Stack all of the data together.
#NOTE: This is not using the full power of EDGE, which allows you to divide up your observations by instrument
vizwl = np.hstack([vizwl, almawl])
vizphot = np.hstack([vizphot, almaflux])
vizphoterr = np.hstack([vizphoterr, almaerr])

#Add the photmetry to the object
red.add_photometry('Literature', vizwl, vizphot, errors = vizphoterr)

#******************************
#De-reddening
#******************************
#Set the Av and uncertainty
Av = .98
Av_unc = 0.00
#Set the de-reddening law. See the EDGE dered docstring for all the available laws
law = 'mathis90_rv3.1'

#De-redden the data
red.dered(Av, Av_unc, law, picklepath, err_prop=1, clob = True)

