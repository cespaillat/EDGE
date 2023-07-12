#!/usr/bin/env python
# Created by Dan Feldman and Connor Robinson for analyzing data from Espaillat Group research models.

#---------------------------------------------IMPORT RELEVANT MODULES--------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy.io import fits
import os
import itertools
import math
import _pickle as cPickle
import pdb
import copy
from glob import glob
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import util
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
commonpath      = edgepath+'/../COMMON/'
datapath        = None #Define this in EDGE.py or use the optional input
figurepath      = None #Define this in EDGE.py or use the optional input
shockpath       = None #Define this in EDGE.py or use the optional input

#---------------------------------------------INDEPENDENT FUNCTIONS----------------------------------------------
# A function is considered independent if it does not reference any other function or class in this module.
# Many of the original functions present here have been moved to util
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

#----------------------------------------------DEPENDENT FUNCTIONS-----------------------------------------------
# A function is considered dependent if it utilizes either the above independent functions, or the classes below.
def look(obs, model=None, jobn=None, save=0, savepath=figurepath, colkeys=None, diskcomb=0, msize=7.0, xlim=[2e-1, 1e4], ylim=None, params=1, leg=1, odustonly = 0, format = '.pdf'):
    """
    Creates a plot of a model and the observations for a given target.

    INPUTS
    model: The object containing the target's model. Should be an instance of the TTS_Model class. This is an optional input.
    obs: The object containing the target's observations. Should be an instance of the TTS_Obs class.
    jobn: The job number you want to use when you save the plot, if different than the one listed in the model.
    save: BOOLEAN -- If 1 (True), will save the plot in a pdf file. If 0 (False), will output to screen.
    savepath: The path that a saved figure(.pdf by default) will be written to. This is defaulted to the hard-coded figurepath at top of this file.
    colkeys: An optional input array of color strings. This can be used to overwrite the normal color order convention. Options include:
             p == purple, r == red, m == magenta, b == blue, c == cyan, l == lime, t == teal, g == green, y == yellow, o == orange,
             k == black, w == brown, v == violet, d == gold, n == pumpkin, e == grape, j == jeans, s == salmon
             If not specified, the default order will be used, and once you run out, we'll have an error. So if you have more than 18
             data types, you'll need to supply the order you wish to use (and which to repeat). Or you can add new colors using html tags
             to the code, and then update this header.
    diskcomb: BOOLEAN -- If 1 (True), will combine outer wall and disk components into one for plotting. If 0 (False), will separate.
    xlim: A list containing the lower and upper x-axis limits, respectively. Has default values.
    ylim: A list containing the lower and upper y-axis limits, respectively. By default it lets matplotlib automatically choose the lower and uppper y-axis limits.
    params: BOOLEAN -- If 1 (True), the parameters for the model will be printed on the plot.
    leg: BOOLEAN -- If 1 (True), the legend will be printed on the plot.
    format: The file format for the saved figure. This is defaulted to pdf.

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

    handles = [] # since some points are potentially plotted individually, this saves the handles for legend to avoid duplicates

    # Plot the spectra first:
    for sind, skey in enumerate(speckeys):
        if 'err' not in obs.spectra[skey].keys():
            handle, = plt.plot(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], 'o', mew=1, markersize=3, \
                     mfc=colors[colkeys[sind]], mec= colors[colkeys[sind]], label=skey)
        else:
            handle = plt.errorbar(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], yerr=obs.spectra[skey]['err'], \
                         mec=colors[colkeys[sind]], fmt='o', mfc=colors[colkeys[sind]], ecolor='silver', ms=1, label=skey)
        handles.append(handle)

    # Next is the photometry:
    for pind, pkey in enumerate(photkeys):
        wl = obs.photometry[pkey]['wl']
        lFl = obs.photometry[pkey]['lFl']

        # reads upper limits from obs file. if none present, generate an empty one for plotting
        if 'ulim' in obs.photometry[pkey].keys():
            ul = obs.photometry[pkey]['ulim']
        else:
            ul = np.zeros(wl.shape)

        # plots upper limits
        handle, = plt.plot(wl[ul==1], lFl[ul==1], 'v', mec=colors[colkeys[pind]], \
            ms=msize, mfc='w', label = pkey, zorder=pind+10)
        
        # plots errorbars or regular points depending if err is present in obs
        if 'err' in obs.photometry[pkey].keys():
            err = obs.photometry[pkey]['err']
            handle = plt.errorbar(wl[ul==0], lFl[ul==0], yerr=err[ul==0], \
                mec=colors[colkeys[pind]], fmt='o', mfc='w', mew=1, ms=msize, \
                ecolor=colors[colkeys[pind]], elinewidth=2, capsize=3, label = pkey, zorder=pind+10)
        else:
            handle, = plt.plot(wl[ul==0], lFl[ul==0], 'o', mfc='w', mec=colors[colkeys[pind]], \
                mew=1, ms=msize, label = pkey, zorder=pind+10)

        handles.append(handle)


    # Now, the model (if a model supplied):
    if model != None:
        if model.components['phot']: # stellar photosphere
            plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere') 

        if model.components['dust']: # optically thin dust
            plt.plot(model.data['wl'], model.data['dust'], ls='--', c='#F80303', linewidth=2.0, label='Opt. Thin Dust')

        if model.components['wall']: # wall for TTS model
            plt.plot(model.data['wl'], model.data['wall']*model.wallH/model.altinh, ls='--', c='#53EB3B', linewidth=2.0, label='Wall')

        if model.components['disk']: # disk for TTS model (full or transitional disk)
            plt.plot(model.data['wl'], model.data['disk'], ls ='--', c = '#f8522c', linewidth = 2.0, label = 'Disk')

        if model.components['iwall']: # inner wall for PTD model (pretransitional disk)
            plt.plot(model.data['wl'], model.data['iwall']*model.wallH/model.iwallH, ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')

        if model.components['idisk']: # inner disk for PTD model
            plt.plot(model.data['wl'], model.data['idisk'], ls ='--', c = '#f8522c', linewidth = 2.0, label = 'Inner Disk')

        if model.components['owall'] and diskcomb == 0: # outer wall for PTD model
            plt.plot(model.data['wl'], model.data['owall']*model.owallH/model.altinh, ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')

        if model.components['odisk']: # outer disk for PTD model
            if diskcomb:
                try:
                    diskflux = model.data['owall']*model.owallH/model.altinh + model.data['odisk']
                except KeyError:
                    print('LOOK: Error, tried to combine outer wall and disk components but one component is missing!')
                else:
                    plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Outer Disk')
            else:
                plt.plot(model.data['wl'], model.data['odisk'], ls ='--', c = '#024747', linewidth = 2.0, label = 'Outer Disk')

        if model.components['scatt']: # scattered light component
            plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')

        if model.components['shock']: # accretion shock
            plt.plot(model.data['WTTS']['wl'], model.data['WTTS']['lFl'], c='b', linewidth=2.0, zorder=1, label='WTTS Photosphere')
            plt.plot(model.data['shock']['wl'], model.data['shock']['lFl'], c=colors['j'], linewidth=2.0, zorder=2, label='MagE')
            plt.plot(model.data['shockLong']['wl'], model.data['shockLong']['lFl'], c=colors['s'], linewidth=2.0, zorder=2, label='Shock Model')

        if model.components['total']: # total flux
            plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')

    # Now, the relevant meta-data:
    if model != None:
        if params:
            if odustonly == False:
                plt.figtext(0.60,0.88,'d2g = '+ str(model.d2g), color='#010000', size='9')
                plt.figtext(0.60,0.85,'Eps = '+ str(model.eps), color='#010000', size='9')
                plt.figtext(0.60,0.82,'Rin = '+ str(model.rin), color='#010000', size='9')
                plt.figtext(0.60,0.79,'Altinh = '+ str(model.wallH), color='#010000', size='9')
                plt.figtext(0.80,0.88,'Alpha = '+ str(model.alpha), color='#010000', size='9')
                plt.figtext(0.80,0.85,'Rout = '+ str(model.rdisk), color='#010000', size='9')
                plt.figtext(0.80,0.82,'Mdot = '+ str(model.mdot), color='#010000', size='9')
                plt.figtext(0.40,0.85,'Amax = '+ str(model.amax), color='#010000', size='9')
                plt.figtext(0.40,0.82,'Amaxb = '+ str(model.amaxb), color='#010000', size='9')
                try:
                    plt.figtext(0.40,0.88,r'M$_{disk}$ = '+ str(round(model.diskmass,5)), color='#010000', size='9')
                except TypeError:
                    plt.figtext(0.40,0.88,r'M$_{disk}$ = '+ model.diskmass, color='#010000', size='9')
                # If we have an outer wall height:
                try:
                    plt.figtext(0.80,0.79,'AltinhOuter = '+ str(model.owallH), color='#010000', size='9')
                except AttributeError:
                    plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.temp), color='#010000', size='9')
                else:
                    plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.itemp), color='#010000', size='9')
                    plt.figtext(0.80,0.76,'OWall Temp = '+ str(model.temp), color='#010000', size='9')
                if model.components['idisk']:
                    plt.figtext(0.60, 0.73, 'IDisk Rout = '+str(model.irdisk), color = '#010000', size = '9')
                    plt.figtext(0.80, 0.73, 'IDisk Jobn = '+str(model.ijobn), color = '#010000', size = '9')

    # Lastly, the remaining parameters to plotting (mostly aesthetics):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim[0], xlim[1])
    if(ylim != None):
        plt.ylim(ylim[0], ylim[1])
    plt.ylabel(r'${\rm \lambda F_{\lambda}\; (erg\; s^{-1}\; cm^{-2})}$')
    plt.xlabel(r'${\rm {\bf \lambda}\; (\mu m)}$')
    plt.title(obs.name.upper())
    if leg:
        plt.legend(handles=handles, loc=3, numpoints = 1,fontsize=9)

    # Should we save or should we plot?
    if save:
        if jobn == None:
            try:
                jobstr      = str(model.jobn).zfill(model.fill)
            except AttributeError:
                plt.savefig(savepath + obs.name.upper() + '_obsdata' + format, dpi=300)
            else:
                plt.savefig(savepath + obs.name.upper() + '_' + jobstr + format, dpi=300)
        else:
            try:
                jobstr      = str(jobn).zfill(model.fill)
            except AttributeError:
                plt.savefig(savepath + obs.name.upper() + '_obsdata' + format, dpi=300)
            else:
                plt.savefig(savepath + obs.name.upper() + '_' + jobstr + format, dpi=300)
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
    targList = glob(dpath+'*')
    targList = [x[len(dpath):] for x in targList]

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
                flist = glob(picklepath+'*')
                flist = [x[len(picklepath):] for x in flist]
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
                flist = glob(picklepath+'*')
                flist = [x[len(picklepath):] for x in flist]
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
                flist = glob(picklepath+'*')
                flist = [x[len(picklepath):] for x in flist]
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
                flist = glob(picklepath+'*')
                flist = [x[len(picklepath):] for x in flist]

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

def loadObs(name, datapath = datapath, red = False):
    '''
    Loads in a fits file saved from the TTS_Obs class and places that information into a TTS_obs object

    INPUTS:
        name: [string] name of the object stored in the fits file

    OPTIONAL INPUTS:
        datapath: [string] Location of the fits file. Default is the datapath

    AUTHOR:
        Connor Robinson, October 19th, 2017
    '''

    #Read in the object
    if red == False:
        HDU = fits.open(datapath+name+'_obs.fits')

    else:
        HDU = fits.open(datapath+name+'_red.fits')
    #Create the empty TTS_Obs object

    if red:
        obj = Red_Obs(name)
    else:
        obj = TTS_Obs(name)

    #Extract the unique instrument keys
    photkeys = list(np.unique(HDU[1].data['instrument']))
    speckeys = list(np.unique(HDU[2].data['instrument']))

    #Add photometry
    for pkey in photkeys:
        obj.add_photometry(pkey, HDU[1].data['wl'][HDU[1].data['instrument'] == pkey], HDU[1].data['lFl'][HDU[1].data['instrument'] == pkey], \
            HDU[1].data['err'][HDU[1].data['instrument'] == pkey], ulim = HDU[1].data['ulim'][HDU[1].data['instrument'] == pkey])

    #Add spectra
    for skey in speckeys:
        obj.add_spectra(skey, HDU[2].data['wl'][HDU[2].data['instrument'] == skey], HDU[2].data['lFl'][HDU[2].data['instrument'] == skey], HDU[2].data['err'][HDU[2].data['instrument'] == skey])

    return obj

def job_file_create(jobnum, path, fill=3, iwall=False, inostruc=False,
imod=False, sample_path = None, image = False, **kwargs):
    """
    Creates a new job file that is used by the D'Alessio Model.

    INPUTS
    jobnum: The job number used to name the output job file.
    path: The path containing the sample job file (if sample_path is not used), and ultimately, the output.
    fill: Pads the output file such that the name will be jobXXX if 3, jobXXXX if 4, etc.
    iwall: BOOLEAN -- if True (1), output will turn off switches so we just run as inner wall.
    imod: BBOOLEAN -- if True, it will run gap_creator to modify the structure of the disk.
    image: BOOLEAN -- if True, it will create a job_file for an image instead of an SED.
    sample_path: The path containing the sample job file. If not set, the job_sample will be searched in path.
    **kwargs: The keywords arguments used to make changes to the sample file. Available
              kwargs include:
        amaxs - maximum grain size in disk
        eps - settling parameter
        ztran - height of transition between big and small grains
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
        inter_r - intervals of radius in which the disk structure will be modified
        rho_deltas - deltas by which the density will be multiplied in the radius intervals provided.
        temp_deltas - deltas for temperature
        epsbig_deltas - deltas for epsilon big
        eps_deltas - deltas for epsilon

              kwargs used only for the image (if image is True):
        wavelength - wavelength (in microns) at which the image will be calculated
        imagetype - Type of image. Can be thick or thin, depending on whether the disk will be optically thin or optically thick at this wavelength

        Some can still be included, such as dust grain compositions. They just aren't
        currently supported. If any supplied kwargs are unused, it will print at the end.

    OUTPUT
    A job file with the name jobXXX, where XXX is the three-string number from 001 - 999. If
    No formal outputs are returned by this function; the file is created in the path directory.
    """
    # If sample_path has not been set, it is assumed to be path
    if sample_path == None:
        sample_path = path
    # Is this a jobfile for an image or an SED?
    if image:
        sample = 'job_image'
    else:
        sample = 'job_sample'

    # First we have to make sure that the job_sample file has been "fixed" for the \r issue:
    os.system("cat " + sample_path + sample + " | tr -d '\r' > " + sample_path + sample + "2")
    os.system("mv " + sample_path + sample + "2 " + sample_path + sample)

    # Next, let's read in the sample job file so we have a template:
    job_file = open(sample_path+sample, 'r')
    fullText = job_file.readlines()     # All text in a list of strings
    job_file.close()

    text = ''.join(fullText)

    if len(fullText) == 0:
        raise ValueError('JOB_FILE_CREATE: job_sample file missing/empty!')

    #Now go through the parameters
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

        #Fix the special case of lamaxb
        if param == 'LAMAXB':
            amaxdict={'100':'100','200':'200','300':'300','400':'400',\
            '500':'500', '600':'600', '700':'700', '800':'800',\
            '900':'900', '1mm':'1000', '2mm':'2000', '3mm':'3000', '4mm':'4000',\
            '5mm':'5000', '6mm':'6000', '7mm':'7000', '8mm':'8000', '9mm':'9000',\
            '1cm':'10000', '1p1cm':'11000', '1p2cm':'12000', '1p3cm':'13000',\
            '1p4cm':'14000', '1p5cm':'15000', '1p6cm':'16000', '1p7cm':'17000',\
            '1p8cm':'18000', '1p9cm':'19000', '2cm':'20000', '2p1cm':'21000',\
            '2p2cm':'22000', '2p3cm':'23000', '2p4cm':'24000', '2p5cm':'25000'}
            paramstr2 = amaxdict[paramstr]

            start = text.find("set lamaxb='") + len("set lamaxb='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + 'amax' + paramstr + text[end:]

            start = text.find("set AMAXB='") + len("set AMAXB='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + paramstr2 + text[end:]

        #Fix the special case of amaxs
        elif param == 'AMAXS':
            amaxdict={'0.05':'0p05', '0.1':'0p1', '0.25':'0p25', '0.5':'0p5',\
            '0.75':'0p75', '1.0':'1p0', '1.25':'1p25', '1.5':'1p5',\
            '1.75':'1p75', '2.0':'2p0', '2.25':'2p25', '2.5':'2p5',\
            '3.0':'3p0', '4.0':'4p0', '5.0':'5p0', '10.0':'10', '100.0':'100'}
            paramstr2 = amaxdict[paramstr]

            start = text.find("set lamaxs='") + len("set lamaxs='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + 'amax' + paramstr2 + text[end:]

            start = text.find("set AMAXS='") + len("set AMAXS='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + paramstr + text[end:]

        #Fix the special case of amaxw
        elif param == 'AMAXW':
            amaxdict={'0.05':'0p05', '0.1':'0p1', '0.25':'0p25', '0.5':'0p5',\
            '0.75':'0p75', '1.0':'1p0', '1.25':'1p25', '1.5':'1p5',\
            '1.75':'1p75', '2.0':'2p0', '2.25':'2p25', '2.5':'2p5',\
            '3.0':'3p0', '4.0':'4p0', '5.0':'5p0', '10.0':'10', '100.0':'100'}
            paramstr2 = amaxdict[paramstr]

            start = text.find("set lamaxw=") + len("set lamaxw=")
            end = start + len(text[start:].split("\n")[0])
            text = text[:start] + "'amax" + paramstr2 + "'" + text[end:]

            start = text.find("set AMAXW=") + len("set AMAXW=")
            end = start + len(text[start:].split("\n")[0])
            text = text[:start] + "'" + paramstr + "'" + text[end:]


        #Fix the special case of temp + Tshock
        elif param == 'TEMP':
            start = text.find('set '+param+"=") + len('set '+param+"=")
            end = start + len(text[start:].split(" #")[0])
            text = text[:start]+paramstr+text[end:]
        #This parameter does not work with the image code yet
        elif param == 'TSHOCK':
            if image:
                print("WARNING: parameter 'TSHOCK' is not yet supported by image code.")
            else:
                start = text.find('set '+param+"=") + len('set '+param+"=")
                end = start + len(text[start:].split(" #")[0])
                text = text[:start]+paramstr+text[end:]
        #Fix the special case of MDOTSTAR (Sometimes it is $MDOT)
        #Also, this parameter does not work with the image code yet
        elif param == 'MDOTSTAR':
            if image:
                print("WARNING: parameter 'MDOTSTAR' is not yet supported by image code.")
            else:
                start = text.find('set '+param+'=') + len('set '+param+'=')
                end = start + len(text[start:].split("#")[0])
                text = text[:start]+"'"+paramstr+"'"+' '+text[end:]
        #Special case of wavelength (only used if creating jobfile for image)
        elif param == 'WAVELENGTH':
            if image:
                start = text.find('set WL=') + len('set WL=')
                end = start + len(text[start:].split(" #")[0])
                text = text[:start]+paramstr+text[end:]
            else:
                print("WARNING: You used the parameter 'wavelength', but it is \
                only used when creating a jobfile for an image.")
        #Special case of imagetype (only used if creating jobfile for image)
        elif param == 'IMAGETYPE':
            if image:
                if (paramstr != 'thin') and (paramstr != 'thick'):
                    raise IOError("JOB_FILE_CREATE: imagetype can only be 'thin' or 'thick'")
                start = text.find("set image='") + len("set image='")
                end = start + len(text[start:].split("'")[0])
                text = text[:start]+paramstr+text[end:]
            else:
                print("WARNING: You used the parameter 'imagetype', but it is \
                only used when creating a jobfile for an image.")

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

            if "set "+param+"='" not in text:
                raise IOError('JOB_FILE_CREATE: parameter '+param+' not found. You might be using an old jobsample.')
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

    if inostruc:
        turnoff = ['IOPA', 'IVIS', 'IIRR']
        for switch in turnoff:
            start = text.find('set '+switch+"='") + len('set '+switch+"='")
            end = start + len(text[start:].split("'")[0])
            text = text[:start] + '0' + text[end:]

    if imod:
        # Turn on gap_creator to modify disk structure
        start = text.find("set IMOD='") + len("set IMOD='")
        end = start + len(text[start:].split("'")[0])
        text = text[:start] + '1' + text[end:]


    outtext = [s + '\n' for s in text.split('\n')]

    # Once all changes have been made, we just create a new job file:
    if image:
        outjob = 'job_image'
    else:
        outjob = 'job'
    string_num  = str(jobnum).zfill(fill)
    newJob      = open(path+outjob+string_num, 'w')
    newJob.writelines(outtext)
    newJob.close()

    # Lastly, check for unused kwargs that may have been misspelled:
    if len(kwargs) != 0:
        print('JOB_FILE_CREATE: Unused kwargs, could be mistakes:')
        print(kwargs.keys())

    return

def job_optthin_create(jobn, path, fill=3, sample_path = None, **kwargs):
    """
    Creates a new optically thin dust job file.

    INPUTS
    jobn: The job number used to name the output job file.
    path: The path containing the sample job file (if sample_path is not used), and ultimately, the output.
    fill: Pads the output file such that the name will be job_optthinXXX if 3, job_optthinXXXX if 4, etc
    sample_path: The path containing the sample job file. If not set, the job_sample will be searched in path.
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

    # If sample_path has not been set, it is assumed to be path
    if sample_path == None:
        sample_path = path
    # First, load in the sample job file for a template:
    job_file = open(sample_path+'job_optthin_sample', 'r')
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

def create_runall(jobstart, jobend, clusterpath, optthin = False, image = False, outpath = '', commonpath = commonpath, fill = 3):
    '''
    create_runall()

    INPUTS:
        jobstart: [int] First job file in grid
        jobsend: [int] Last job file in grid

    OPTIONAL INPUTS:
        optthin: [Boolean] Set to True for optically thin dust models.
        image: [Boolean] Set to True for images (If optthin is also True, this will not have any effect).
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
    elif image:
        start = text.find('job%0')
        end = start+len('job%0')
        text = text[:start]+'job_image%0'+text[end:]

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

    TODO: Needs to be updated to accept ulim as a list
    """

    #Make observation files backwards compatible
    if 'phot_dens' not in dir(obj):
        obj.phot_dens = 0.0
    if 'spec_dens' not in dir(obj):
        obj.spec_dens = {}

    #First check to see if there is a total component
    if 'total' not in model.data:
        if verbose:
            print("MODEL_RCHI2: Model "+model.jobn+" does not have 'total' values, returning...")
        return -1

    #If a single set of data is given as a string for obsNeglect, turn it into a list
    if type(obsNeglect) == 'str':
        obsNeglect = [obsNeglect]

    #Get number of spectra keys in the neglected objects
    specneglect = np.sum([key in obj.spectra.keys() for key in obsNeglect])

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
                    # We set nans as 10%
                    obs_err[np.isnan(obs_err)] = 0.1 * obs_flux[np.isnan(obs_err)]
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
        #Convert any elements that are not arrays into arrays for=
        wavelength = np.hstack(wavelength)
        flux = np.hstack(flux)
        errs = np.hstack(errs)

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
        modelFlux      = np.power(10,np.interp(np.log10(wavelength), np.log10(model.data['wl']), np.log10(model.data['total'])))

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
    if len(obj.spectra.keys())-specneglect > 0: # If there are any spectra
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
                ws = np.sqrt(obj.spec_dens[specKey] / obj.phot_dens)
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
        modelFluxS     = np.power(10,np.interp(np.log10(wavelengthS), np.log10(model.data['wl']), np.log10(model.data['total'])))

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

    TODO: Needs to be updated to accept ulim as a list
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
            normVal, normErr = (util.linearInterp(normWL, dataDict['wl'][normInd-1], dataDict['wl'][normInd],
                                             dataDict['lFl'][normInd-1], dataDict['lFl'][normInd],
                                             dataDict['err'][normInd-1], dataDict['err'][normInd]))
        else:
            normVal, normErr = (util.linearInterp(normWL, dataDict['wl'][normInd-1], dataDict['wl'][normInd],
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


def binSpectra(obs, speckeys=[], ppbin=2):
    """
    Bin all the spectra in the obs object corresponding to the supplied
    keys.

    INPUTS
    obs: The observations object, which should be an instance if TTS_Obs or Red_Obs.
    speckeys: The keys corresponding to the spectra that should be binned.
    ppbin: The number of points in a given bin.

    OUTPUT
    Though no output is explicitly given, the binned spectra are saved to the observations object.
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


def temp_structure(model, figure_path=None, bw=False,xlim=[],ylim=[]):
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
    if xlim:
        ax.set_xlim([xlim[0],xlim[1]])
    if ylim:
        ax.set_ylim([ylim[0],ylim[1]])

    # save figure and close it
    plt.savefig(figname, dpi=200, bbox_inches='tight')
    plt.close()
    model.close()

    return


#---------------------------------------------------CLASSES------------------------------------------------------
class TTS_Model(object):
    """
    Contains all the data and meta-data for a TTS Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.

    ATTRIBUTES
    name: Name of the object (e.g., CVSO109, V410Xray-2, ZZ_Tau, etc.).
    jobn: The job number corresponding to this model.
    mstar: Star's mass (Msun).
    tstar: Star's effective temperature (K).
    rstar: Star's radius (Rsun).
    dist: Distance to the star (pc).
    mdot: Mass accretion rate (Msun/yr).
    mdotstar: Mass accretion rate onto the star. Usually same as mdot but not necessarily.
    alpha: Alpha parameter (from the viscous alpha disk model).
    mui: Inclination of the system.
    rdisk: The outer radius of the disk (au).
    d2g: dust to gas mass ratio of the disk.
    amax: Maximum grain size in the disk atmoshpere (microns).
    amaxw: Maximum grain size in the wall (microns).
    eps: The epsilon parameter, i.e., the amount of dust settling in the disk.
    ztran: height of transition between big and small grains (in hydrostatic scale heights).
    tshock: The temperature of the shock at the stellar photosphere (K).
    temp: The temperature at the inner wall (K).
    altinh: Height of the wall (in hydrostatic scale heights).
    zwall: Height of the wall (au).
    wlcut_an: Cut in wavelengths for thermal SED (microns)
    wlcut_sc: Cut in wavelengths for scattering (microns)
    nsilcomp: Number of silicate compounds.
    siltotab: Total silicate abundance. (DEPRECATED)
    amorf_ol: Olivine fractional abundance.
    amorf_py: Pyroxene fractional abundance.
    forsteri: Forsterite fractional abundance.
    enstatit: Enstatite fractional abundance.
    silab: Abundance of silicates.
    grafab: Abundance of graphite.
    iceab: Abundance of water ice.
    rin: Inner radius (au).
    rc: Critial radius for tapered edge (au).
    gamma: Index of tapered edge.
    dpath: Path where the data files are located.
    fill: How many numbers used in the model files (4 = name_XXXX.fits).
    data: The data for each component inside the model.
    extcorr: The self-extinction correction. If not carried out, saved as None.
    new: Whether or not the model was made with the newer version of collate.py.
    newwall: The flux of an inner wall with a higher/lower altinh value.
    wallH: The inner wall height used by the look() function in plotting.
    filters: Filters used to calculate synthetic fluxes.
    synthFlux: Wavelengths and syntethic fluxes.

    Structure of disk model. Arrays [nz,nrad]:
    radii_struc: Radii in au.
    z_struc: heights in au.
    T_struc: Temperature in K.
    p_struc: pressure in cgs.
    rho_struc: density in g/cm3
    epsbig_struc: epsbig
    eps_struc: eps

    METHODS
    __init__: Initializes an instance of the class, and loads in the relevant metadata.
    dataInit: DEPRECATED.
    calc_total: Calculates the "total" (combined) flux based on which components you want, then loads it into
                the data attribute under the key 'total'.
    calc_filters: Calculates synthetic fluxes for different instruments.
    blueExcessModel:
    struc_plot: Makes plots of 2D structure of the disk model.
    """

    def __init__(self, name, jobn, dpath=datapath, fill=3, verbose=False):
        """
        Initializes instances of this class and loads the relevant data into attributes.

        Also initializs data attributes using nested dictionaries:
        - wl is the wavelength (corresponding to all three flux arrays).
        - Phot is the stellar photosphere emission.
        - wall is the flux from the inner wall.
        - Disk is the emission from the angle file.
        - Scatt is the scattered light emission.
        - Loads in self-extinction array if available.
        - Loads structure of disk.

        INPUTS
        name: Name of the object being modeled. Must match naming convention used for models.
        jobn: Job number corresponding to the model being loaded into the object. Again, must match convention.
        full_trans: BOOLEAN -- if 1 (True) will load data as a full or transitional disk. If 0 (False), as a pre-trans. disk.
        fill: How many numbers the input model file has (jobXXX vs. jobXXXX, etc.)
        verbose: BOOLEAN -- if 1 (True), will print out warnings about missing components.
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
        self.amaxb      = header['AMAXB']
        self.eps        = header['EPS']
        self.tshock     = header['TSHOCK']
        self.temp       = header['TEMP']
        self.altinh     = header['ALTINH']
        self.wlcut_an   = header['WLCUT_AN']
        self.wlcut_sc   = header['WLCUT_SC']
        self.nsilcomp   = header['NSILCOMP']
        #self.siltotab   = header['SILTOTAB']
        self.amorf_ol   = header['AMORF_OL']
        self.amorf_py   = header['AMORF_PY']
        self.forsteri   = header['FORSTERI']
        self.enstatit   = header['ENSTATIT']
        self.rin        = header['RIN']
        self.diskmass   = header['DISKMASS']
        self.dpath      = dpath
        self.fill       = fill
        self.extcorr    = None
        self.filters    = {}
        self.synthFlux= {}
        # Try retrieving newer parameters, for backwards compatibility
        try:
            self.mdotstar = header['MDOTSTAR']
        except KeyError:
            self.mdotstar = self.mdot
        try:
            self.amaxw = header['AMAXW']
        except:
            print('WARNING: AMAXW not found. This is probably an old collated '+
            'model. Setting it to AMAXS')
            self.amaxw = header['AMAXS']
        try:
            self.ztran = header['ZTRAN']
        except:
            print('WARNING: ZTRAN not found. This is probably an old collated '+
            'model.')
        try:
            self.zwall = header['ZWALL']
        except:
            print('WARNING: ZWALL not found. This is probably an old collated '+
            'model.')
        try:
            self.d2g = header['D2G']
        except:
            print('WARNING: D2G not found. This is probably an old collated '+
            'model. Setting it to NaN.')
            self.d2g = np.nan
        try:
            self.rc = header['RC']
            self.gamma = header['GAMMA']
        except:
            print('WARNING: RC and GAMMA not found. This is probably an old '+
            'collated model.')
        try:
            self.silab = header['SILAB']
            self.grafab = header['GRAFAB']
            self.iceab = header['ICEAB']
        except:
            print('WARNING: SILAB, GRAFAB, and ICEAB not found. This is '+
            'probably an old collated model. Setting them to NaN.')
            self.silab = np.nan
            self.grafab = np.nan
            self.iceab = np.nan

        # Load SED in nested dictionaries
        # The new Python version of collate flips array indices, so must identify which collate.py was used:
        if 'EXTAXIS' in header.keys() or 'NOEXT' in header.keys():
            self.new = 1
        else:
            self.new = 0

        if self.new:
            # We will load in the components piecemeal based on the axes present in the header.
            # First though, we initialize with the wavelength array, since it's always present:
            self.data = {'wl': HDUlist[0].data[header['WLAXIS'],:]}

            # Now we can loop through the remaining possibilities:
            if 'PHOTAXIS' in header.keys():
                self.data['phot'] = HDUlist[0].data[header['PHOTAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No photosphere data found for ' + self.name)
            if 'WALLAXIS' in header.keys():
                self.data['wall'] = HDUlist[0].data[header['WALLAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No outer wall data found for ' + self.name)
            if 'ANGAXIS' in header.keys():
                self.data['disk'] = HDUlist[0].data[header['ANGAXIS'],:]
            else:
                if verbose:
                    print('DATAINIT: Warning: No outer disk data found for ' + self.name)
            # Remaining components are not always (or almost always) present, so no warning given if missing!
            if 'SCATAXIS' in header.keys():
                self.data['scatt'] = HDUlist[0].data[header['SCATAXIS'],:]
                negScatt = np.where(self.data['scatt'] < 0.0)[0]
                if len(negScatt) > 0:
                    print('DATAINIT: WARNING: Some of your scattered light values are negative!')
            if 'EXTAXIS' in header.keys():
                self.extcorr       = HDUlist[0].data[header['EXTAXIS'],:]
        else:
            self.data = {'wl': HDUlist[0].data[:,0], 'phot': HDUlist[0].data[:,1], 'wall': HDUlist[0].data[:,2], \
                         'disk': HDUlist[0].data[:,3]}


        # Load structure of the disk
        # If there are more than 2 extensions, then the structure must be in the fits file
        try:
            if header['EXTS'] >= 2:
                self.radii_struc = HDUlist[header['RAD_EXT']-1].data
                self.z_struc = HDUlist[header['Z_EXT']-1].data
                self.T_struc = HDUlist[header['T_EXT']-1].data
                self.p_struc = HDUlist[header['P_EXT']-1].data
                self.rho_struc = HDUlist[header['RHO_EXT']-1].data
                self.epsbig_struc = HDUlist[header['EPSB_EXT']-1].data
                self.eps_struc = HDUlist[header['EPS_EXT']-1].data
        except:
            print('WARNING: Structure not found, this is probably an old model.')
        HDUlist.close()
        return

    def dataInit(self, verbose=1):
        """
        DEPRECATED. Done automatically by __init__
        """
        print('WARNING: This is now done automatically when initializing the object. This method will disappear in the future.')
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

        if np.all(self.extcorr) != None:
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
                self.newwall = self.data['wall'] * altinh
                totFlux       = totFlux + self.newwall     # Note: if save=1, will save wall w/ the original altinh.
                self.wallH    = self.altinh * altinh
            else:
                totFlux       = totFlux + self.data['wall']
                self.wallH    = self.altinh                 # Redundancy for plotting purposes.
                # If we tried changing altinh but want to now plot original, deleting the "newwall" attribute from before.
                try:
                    del self.newwall
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

        #Add flags to the model to signify which components have been added together
        self.components = {'total':1, 'phot':phot, 'wall':wall, 'disk':disk, 'dust':dust, 'scatt':scatt, 'iwall':0, 'idisk':0, 'owall':0, 'odisk':0, 'shock':0}

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
                outputTable[:, colNum] = self.data['wall']
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
    def calc_filters(self,filterspath=commonpath+'Filters/',obj='',verbose=1):
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
                UCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_V-1_km.txt')
                BCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_V-1_km.txt')
                VCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_V-1_km.txt')
                RCVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_R-1_km.txt')
                ICVSOfilter        = np.loadtxt(filterspath+'Johnson-Cousins_filters/Bessel_I-1_km.txt')
                self.filters[instrKey] = {'f0.37': {'wl':UCVSOfilter[:,0]*0.001,'trans':UCVSOfilter[:,1]},\
                                        'f0.44': {'wl':BCVSOfilter[:,0]*0.001,'trans':BCVSOfilter[:,1]},\
                                        'f0.55': {'wl':VCVSOfilter[:,0]*0.001,'trans':VCVSOfilter[:,1]},\
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
                if verbose:
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
            self.newwall = np.interp(self.data['wl'], oldWavelength, self.newwall)
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

        self.components['shock'] = 1

        return

    def struc_plot(self, temp = True, rho = True, eps = False, epsbig = False, pres = False, logscale = True,
    xlim=[], ylim=[], temp_levels=None, rho_levels=None, eps_levels=None, epsbig_levels=None, pres_levels=None,
    snowlines=True, plothscale = True, label=None):
        '''
        Makes plots of 2D structure (temperature, density, pressure, epsilon,
        and epsilonbig) of the disk model.

        OPTIONAL INPUTS:
        - temp, rho, eps, epsbig, pres: set to True if you want to plot the temperature, density,
          epsilon, epsilonbig, and pressure, respectively. DEFAULT: just temp and rho.
        - ***_levels: Contour levels for the plots.
        - snowlines: If True, it will plot the contours for the water, CO, and N2 snowlines in
          the temperature plots.
        - xlim, ylim: Lists containing the lower and upper x-axis and y-axis limits, respectively.
          DEFAULT: minimum and maximum values of array to be plotted.
        - logscale: If True, plots will be in log-log scale. DEFAULT: True.
        - label: Label to be used in the output file of the plots
        '''
        cmap = cm.nipy_spectral
        if label == None:
            label = str(self.jobn)

        npoints = self.z_struc.size

        # Create the grid of points
        if logscale:
            radii_grid = np.log10(self.radii_struc.reshape(npoints,1))
            z_grid = np.log10(self.z_struc.reshape(npoints,1)+1e-20)
        else:
            radii_grid = self.radii_struc.reshape(npoints,1)
            z_grid = self.z_struc.reshape(npoints,1)
        points = np.concatenate((radii_grid,z_grid),axis = 1)

        # Limits of the plots
        if xlim:
            rmin = xlim[0]
            rmax = xlim[1]
        else:
            rmin = np.min(self.radii_struc)
            rmax = np.max(self.radii_struc)

        if ylim:
            zmin = ylim[0]
            zmax = ylim[1]
        else:
            zmin = np.min(self.z_struc)
            zmax = np.max(self.z_struc)

        if logscale:
            zmin = 0.01
            rmin = np.log10(rmin)
            rmax = np.log10(rmax)
            zmin = np.log10(zmin)
            zmax = np.log10(zmax)

        aspectratio = (rmax - rmin) / (zmax-zmin)

        grid_x, grid_y = np.mgrid[rmin:rmax:500j, zmin:zmax:500j]

        # Paremeters to plot
        params = {}
        labels = {}
        titles = {}
        levels = {}
        vmins = {}
        vmaxs = {}
        if temp:
            params['temp'] = self.T_struc # Values to plot
            labels['temp'] = '_temp' # String to include in name of output file
            titles['temp'] = 'Temperature (K)' # Title of plot
            if temp_levels == None: # Define default contour levels
                temp_levels = [10,20,40,100,300,600,1400]
                # Define default minimum and maximum values
                temp_vmin = np.min(self.T_struc)
                temp_vmax = np.max(self.T_struc)
            else:
                temp_vmin = np.min(temp_levels)
                temp_vmax = np.max(temp_levels)
            levels['temp'] = temp_levels
            vmins['temp'] = np.log10(temp_vmin)
            vmaxs['temp'] = np.log10(temp_vmax)
        if rho:
            params['rho'] = self.rho_struc
            labels['rho'] = '_rho'
            titles['rho'] = r'Density (g cm$^{-3}$)'
            if rho_levels == None:
                rho_levels = [1e-16,1e-14,1e-12,1e-10,1e-8]
                rho_vmin = np.min(self.rho_struc)
                rho_vmax = np.max(self.rho_struc)
            else:
                rho_vmin = np.min(rho_levels)
                rho_vmax = np.max(rho_levels)
            levels['rho'] = rho_levels
            vmins['rho'] = np.log10(rho_vmin)
            vmaxs['rho'] = np.log10(rho_vmax)
        if pres:
            params['pres'] = self.p_struc
            labels['pres'] = '_pressure'
            titles['pres'] = 'Pressure (cgs)'
            if pres_levels == None:
                pres_levels = [1e-7,1e-4,1e-1,1e2,1e4]
                pres_vmin = np.min(self.p_struc)
                pres_vmax = np.max(self.p_struc)
            else:
                pres_vmin = np.min(pres_levels)
                pres_vmax = np.max(pres_levels)
            levels['pres'] = pres_levels
            vmins['pres'] = np.log10(pres_vmin)
            vmaxs['pres'] = np.log10(pres_vmax)
        if eps:
            params['eps'] = self.eps_struc + 1e-8 # To avoid 0 and infs
            labels['eps'] = '_eps'
            titles['eps'] = 'Epsilon (small grains)'
            if eps_levels == None:
                eps_levels = [np.max(self.eps_struc)]
            levels['eps'] = eps_levels
            eps_vmin = np.max([np.min(self.eps_struc),1e-4])
            eps_vmax = np.max(self.eps_struc)
            vmins['eps'] = np.log10(eps_vmin)
            vmaxs['eps'] = np.log10(eps_vmax)
        if epsbig:
            params['epsbig'] = self.epsbig_struc + 1e-8 # To avoid 0 and infs
            labels['epsbig'] = '_epsbig'
            titles['epsbig'] = 'Epsilon (large grains)'
            if epsbig_levels == None:
                epsbig_levels = [np.max(self.epsbig_struc)]
            levels['epsbig'] = epsbig_levels
            epsbig_vmin = np.max([np.min(self.epsbig_struc),1e-4])
            epsbig_vmax = np.max(self.epsbig_struc)
            vmins['epsbig'] = np.log10(epsbig_vmin)
            vmaxs['epsbig'] = np.log10(epsbig_vmax)

        # Plots of each parameter
        for param in params.keys():
            name = label + labels[param]
            values = (np.log10(params[param])).reshape(npoints)
            grid_z1 = griddata(points,values,(grid_x, grid_y),method = 'linear')
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            # Plot of contours
            CS1 = ax1.contour(grid_x, grid_y, grid_z1, np.log10(levels[param]),linestyles='solid',colors=('black'))
            # Labels of contours
            fmt = {}
            for l,s in zip(CS1.levels,map(str,levels[param])):
                fmt[l] = s
            plt.clabel(CS1,inline=1,inline_spacing=15, fmt=fmt, fontsize=11,linewidths=2)
            # Plot snowlines if we are interested
            if (param == 'temp') & snowlines: # Plot important snowlines
                CSH2O = ax1.contour(grid_x, grid_y, grid_z1, np.log10([150.]),linestyles='dashed',colors=('red'))
                plt.clabel(CSH2O,inline=1,inline_spacing=15, fmt=r'H$_2$O (180 K)', colors=('red'), fontsize=11,linewidths=2)
                CSCO = ax1.contour(grid_x, grid_y, grid_z1, np.log10([26.]),linestyles='dashed',colors=('yellow'))
                plt.clabel(CSCO,inline=1,inline_spacing=15, fmt=r'CO (26 K)', colors=('yellow'), fontsize=11,linewidths=2)
                CSN2 = ax1.contour(grid_x, grid_y, grid_z1, np.log10([22.]),linestyles='dashed',colors=('green'))
                plt.clabel(CSN2,inline=1,inline_spacing=15, fmt=r'N$_2$ (22 K)', colors=('green'), fontsize=11,linewidths=2)
            plt.tick_params(axis='y', labelsize=11)
            plt.tick_params(axis='x', labelsize=11)
            ax1.set_xlabel('R (au)',fontsize=11)
            ax1.set_ylabel('z (au)',fontsize=11)
            ax1.set_title(titles[param],fontsize=11)
            # Tick marks
            if logscale:
                thickplaces=[np.log10(i*j) for i in [0.01,0.1,1.0,10.0,100.0] for j in [1,2,3,4,5,6,7,8,9] ]
                thickmarks=[]
                for i in thickplaces:
                    if (math.fabs(math.fmod(i,1.0))==0.0):
                        thickmarks.append(str(10**i))
                    else:
                        thickmarks.append('')
                plt.xticks(thickplaces,thickmarks)
                plt.yticks(thickplaces,thickmarks)
                name = name +'_log'
            else:
                ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
                ax1.yaxis.set_minor_locator(ticker.MultipleLocator(1))
            # Image
            im1 = ax1.imshow(grid_z1.T, extent=(rmin,rmax,zmin,zmax), aspect=aspectratio,origin='lower',vmin=vmins[param],vmax=vmaxs[param],cmap=cmap)
            fig.subplots_adjust(left=0.2)
            # Color bar
            cbar=fig.colorbar(im1,label='log('+titles[param]+')')

            plt.savefig('struc_'+name+'.pdf', dpi=300, facecolor='w', edgecolor='w',
                          orientation='landscape', papertype=None, format=None,
                          transparent=False, bbox_inches='tight', pad_inches=0.01,
                          frameon=None)
            plt.close(fig)

        return

class PTD_Model(TTS_Model):
    """
    Contains all the data and meta-data for a PTD Model from the D'Alessio et al. 2006 models. The input
    will come from fits files that are created via Connor's collate.py.

    ATTRIBUTES
    name: Name of the object (e.g., CVSO109, V410Xray-2, ZZ_Tau, etc.).
    jobn: The job number corresponding to this model.
    mstar: Star's mass (Msun).
    tstar: Star's effective temperature (K).
    rstar: Star's radius (Rsun).
    dist: Distance to the star (pc).
    mdot: Mass accretion rate (Msun/yr).
    mdotstar: Mass accretion rate onto the star. Usually same as mdot but not necessarily.
    alpha: Alpha parameter (from the viscous alpha disk model).
    mui: Inclination of the system.
    rdisk: The outer radius of the disk (au).
    d2g: dust to gas mass ratio of the disk.
    amax: Maximum grain size in the disk atmoshpere (microns).
    amaxw: Maximum grain size in the wall (microns).
    eps: The epsilon parameter, i.e., the amount of dust settling in the disk.
    ztran: height of transition between big and small grains (in hydrostatic scale heights).
    tshock: The temperature of the shock at the stellar photosphere (K).
    temp: The temperature at the inner wall (K).
    itemp: The temperature of the inner wall component of the model.
    altinh: Height of the wall (in hydrostatic scale heights).
    zwall: Height of the wall (au).
    wlcut_an: Cut in wavelengths for thermal SED (microns)
    wlcut_sc: Cut in wavelengths for scattering (microns)
    nsilcomp: Number of silicate compounds.
    siltotab: Total silicate abundance. (DEPRECATED)
    amorf_ol: Olivine fractional abundance.
    amorf_py: Pyroxene fractional abundance.
    forsteri: Forsterite fractional abundance.
    enstatit: Enstatite fractional abundance.
    silab: Abundance of silicates.
    grafab: Abundance of graphite.
    iceab: Abundance of water ice.
    rin: Inner radius (au).
    rc: Critial radius for tapered edge (au).
    gamma: Index of tapered edge.
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
    calc_filters: Calculates synthetic fluxes for different instruments.
    blueExcessModel:
    struc_plot: Makes plots of 2D structure of the disk model.
    """

    def dataInit(self, altname=None, jobw=None, fillWall=3, wallpath = None, verbose =1, **searchKwargs):
        """
        Initializes data attributes for inner wall and inner disk,
        and renames outer wall and outer disk:
        - iwall is the flux from the inner wall.
        - idisk is the emission from the inner disk.
        - owall is the flux from the outer wall.
        - odisk is the emission from the outer wall.

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
        if altname == None:
            altname = self.name
        if wallpath == None:
            wallpath  = self.dpath

        if jobw == None and len(searchKwargs) == 0:
            raise IOError('DATAINIT: You must enter either a job number or kwargs to match or search for an inner wall.')

        if jobw != None:
            # If jobw is an integer, make into a string:
            jobw = str(jobw).zfill(fillWall)
            # The case in which you supplied the job number of the inner wall:
            fitsname  = wallpath + altname + '_' + jobw + '.fits'
            HDUwall   = fits.open(fitsname)
        else:
            # When doing the searchJobs() call, use **searchKwargs to pass that as the keyword arguments to searchJobs!
            match = searchJobs(altname, dpath=wallpath, **searchKwargs)
            if len(match) == 0:
                raise IOError('DATAINIT: No inner wall model matches these parameters!')
            elif len(match) > 1:
                raise IOError('DATAINIT: Multiple inner wall models match. Do not know which one to pick.')
            fitsname = wallpath + altname + '_' + match[0] + '.fits'
            HDUwall  = fits.open(fitsname)

        # Make sure the inner wall job you supplied is, in fact, an inner wall.
        if 'WALLAXIS' not in HDUwall[0].header.keys():
            raise IOError('DATAINIT: Job you supplied for inner wall does not '+
            'have a wall!')

        stringNum     = str(self.jobn).zfill(self.fill)

        # Define the inner wall height.
        self.iwallH      = HDUwall[0].header['ALTINH']
        self.itemp       = HDUwall[0].header['TEMP']
        self.ijobn       = HDUwall[0].header['JOBNUM']
        self.insilcomp   = HDUwall[0].header['NSILCOMP']
        #self.isiltotab   = HDUwall[0].header['SILTOTAB']
        self.iamorf_ol   = HDUwall[0].header['AMORF_OL']
        self.iamorf_py   = HDUwall[0].header['AMORF_PY']
        self.iforsteri   = HDUwall[0].header['FORSTERI']
        self.ienstatit   = HDUwall[0].header['ENSTATIT']
        self.irin        = HDUwall[0].header['RIN']
        try:
            self.iamaxw  = HDUwall[0].header['AMAXW']
        except:
            print('WARNING: AMAXW not found for inner wall. This is probably '+
            'an old collated model. Setting it to AMAXS')
            self.iamaxw  = HDUwall[0].header['AMAXS']
        try:
            self.izwall  = HDUwall[0].header['ZWALL']
        except:
            print('WARNING: ZWALL not found for inner wall. This is probably '+
            'an old collated model.')

        # Correct for self extinction:
        if self.extcorr != None:
            iwallFcorr= HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]*np.exp(-1*self.extcorr)
        else:
            print('DATAINIT: WARNING! No extinction correction can be made for job ' + str(self.jobn)+'!')
            iwallFcorr= HDUwall[0].data[HDUwall[0].header['WALLAXIS'],:]
        self.data['iwall'] = iwallFcorr

        #If there is an inner disk, add that as well.
        if 'ANGAXIS' in HDUwall[0].header.keys():
            if 'EXTAXIS' in HDUwall[0].header.keys():
                idiskFcorr= HDUwall[0].data[HDUwall[0].header['ANGAXIS'],:]*np.exp(-1*HDUdata[0].data[header['EXTAXIS'],:])
            else:
                idiskFcorr= HDUwall[0].data[HDUwall[0].header['ANGAXIS'],:]
            self.data['idisk'] = idiskFcorr
            #Add information about the disk
            self.ialpha      = HDUwall[0].header['ALPHA']
            self.irdisk      = HDUwall[0].header['RDISK']
            self.ieps        = HDUwall[0].header['EPS']
            self.iamax       = HDUwall[0].header['AMAXS']
            self.iamaxb      = HDUwall[0].header['AMAXB']

        # We change the keys for the outer wall and outer disk
        # Outer wall
        if 'wall' in self.data.keys():
            self.data['owall'] = self.data['wall']
        else:
            print('DATAINIT: Warning: No outer wall data found for ' + self.name)
        # Outer disk
        if 'disk' in self.data.keys():
            self.data['odisk'] = self.data['disk']
        else:
            print('DATAINIT: Warning: No outer disk data found for ' + self.name)

        HDUwall.close()
        return

    @keyErrHandle
    def calc_total(self, phot=1, iwall=1, idisk=1, owall=1, odisk = 1, dust=0, scatt = 0, verbose=1, dust_fill=3, altInner=None, altOuter=None, save=0, OTDpath=None):
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
        if scatt:
            if verbose:
                print('CALC_TOTAL: Adding scattered light component to the total flux.')
            totFlux     = totFlux + self.data['scatt']
            componentNumber += 1

        # Add the total flux array to the data dictionary attribute:
        if verbose:
            print('CALC_TOTAL: Total flux calculated. Adding to the data structure.')
        self.data['total'] = totFlux
        componentNumber += 1

        #Add flags to the model to signify which components have been added together
        self.components = {'total':1, 'phot':phot, 'iwall':iwall, 'idisk':idisk, 'owall':owall, 'odisk':odisk, 'dust':dust, 'scatt':scatt, 'wall':0, 'disk':0, 'shock':0}

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
    Contains all the observational data for a given target system. Allows you to create a fits file with the data, so it can
    be reloaded in at a future time without the need to re-initialize the object. However, to create the object fits file, you will
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
    saveObs: Saves the object as a fits to be reloaded later.
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
        self.spec_dens = {}
        self.phot_dens = 0.0

    def add_spectra(self, scope, wlarr, fluxarr, errors=None, py2 = False, overwrite = 1):
        """
        Adds an entry to the spectra attribute.

        INPUTS
        scope: The telescope or instrument that the spectrum was taken with.
        wlarr: The wavelenth array of the data. Should be in microns. Note: this is not checked.
        fluxarr: The flux array of the data. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        """

        # Check if the telescope data already exists in the data file:
        if scope in self.spectra.keys() and overwrite == False:
            print('ADD_SPECTRA: Warning! This will overwrite current entry!')
            tries               = 1
            while tries <= 5:                                           # Give user 5 chances to choose if overwrite data or not
                if py2:
                    proceed         = raw_input('Proceed? (Y/N): ')         # Prompt and collect manual answer - requires Y,N,Yes,No (not case sensitive)
                else:
                    proceed         = input('Proceed? (Y/N): ')
                if proceed.upper() == 'Y' or proceed.upper() == 'YES':  # If Y or Yes, overwrite file, then break out of loop
                    print('ADD_SPECTRA: Replacing entry.')
                    if np.all(errors == None):
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
            if np.all(errors) == None:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr}
            else:
                self.spectra[scope] = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
        return

    def add_photometry(self, scope, wlarr, fluxarr, errors=None, ulim=None, verbose = 1, py2 = False):
        """
        Adds an entry to the photometry attribute.

        INPUTS
        scope: The telescope or instrument that the photometry was taken with.
        wlarr: The wavelength array of the data. Can also just be one value if an individual point. Should be in microns. Note: this is not checked.
        fluxarr: The flux array corresponding to the data. Can also just be one value if an individual point. Should be in erg s-1 cm-2. Note: this is not checked.
        errors: (optional) The array of flux errors. Should be in erg s-1 cm-2. If None (default), will not add.
        ulim: The upper limit array (booleans) of the data, indicating which data points are upper limits. Can also just be one value if an individual point.

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
        if type(ulim) == list:
            ulim = np.array(ulim)
        if type(errors) == list:
            errors = np.array(errors)
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
                    if np.all(ulim == None):
                        if errors == None:
                            self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}
                        else:
                            self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
                    else:
                        self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'ulim': ulim, 'err': errors}   #Add ulim if it is given
                    break
                elif proceed.upper() == 'N' or proceed.upper() == 'NO':     # If N or No, do not overwrite data and return
                    print('ADD_PHOTOMETRY: Will not replace entry. Returning now.')
                    return
                else:
                    tries           = tries + 1                             # If something else, lets you try again
            else:
                raise IOError('You did not enter the correct Y/N response. Returning without replacing.')   # If you enter bad response too many times, raise error.
        else:
            if np.all(ulim == None):
                if np.all(errors == None):
                    self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr}     # If not an overwrite, writes data to the object's photometry attribute dictionary.
                else:
                    self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'err': errors}
            else:
                self.photometry[scope]  = {'wl': wlarr, 'lFl': fluxarr, 'ulim': ulim, 'err': errors}   # Add ulim if it is given
        # We reset the attribute phot_dens, since with new photometry it should be recalculated
        self.phot_dens = 0.0

        return

    def SPPickle(self, picklepath, overwrite = False, fill = 3):
        """
        Saves the object as a pickle. Damn it Jim, I'm a doctor not a pickle farmer!

        WARNING: If you reload the module BEFORE you save the observations as a pickle, this will NOT work! I'm not
        sure how to go about fixing this issue, so just be aware of this.

        INPUTS
        picklepath: The path where you will save the pickle. I recommend datapath for simplicity.
        overwrite: boolean, if set to True, will overwrite the old pickle
        fill: How many numbers used in the model files (4 = name_XXXX.fits).

        OUTPUT:
        A pickle file of the name [self.name]_obs.pkl in the directory provided in picklepath.
        """

        # Check whether or not the pickle already exists:
        pathlist = glob(picklepath+'*')
        pathlist = [x[len(picklepath):] for x in pathlist]
        outname         = self.name + '_obs.pkl'
        count           = 1
        while 1:
            if outname in pathlist and overwrite == False:
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

    def saveObs(self, datapath=datapath, overwrite = 1, make_csv = False, dered = True,\
        Av = None, extlaw = None, Mstar = None, Mref = None, Rstar = None, \
        Rref = None, Tstar = None, Tref = None, dist = None, dref = None):
        """
        Saves a TTS_Obs object as a fits file (Replacing pickle files)

        INPUTS:
            None

        OPTIONAL INPUTS:
            datapath:[string] Path where the data will be saved. Default is datapath
            overwrite:[boolean] Overwrites existing files if true
            Av: [float] Extinction (Av)
            extlaw: [string] Law used to de-redden data
            make_csv: [boolean] Makes two .csv files containing photometry and spectra if true
            dered: [boolean] True if the data has been dereddened. Default here is True, since this
                   is working with the TTS_Obs class

            Mtar: [float] Mass of the star in Msun
            Mref: [string] Reference for Mstar

            Rstar: [float] Radius of the star in Rstar
            Rref: [string] Reference for Rstar

            Tstar: [float] Temperature of the star in K
            Tref: [string] Reference for Tstar

            dist: [float] Distance to the star in pc
            dref: [string] Reference for dist

        OUTPUT:
            A fits file with there extensions.
            The primary extension (extension 0) only has information in the header about the target
            The second extension (extension 1) contains photometry of the object in a binary fits table
            The third extension (extension 2) contains spectra of the object in a binary fits table

            Each table contains data points, errors, and the associated instrument. Extension 1 also contains
            if the object is an upper limit

        Author:
            Connor Robinson, October 19th, 2017
        """
        photkeys = list(self.photometry.keys())
        speckeys = list(self.spectra.keys())

        if make_csv == False:
            #Break the photometry down into something easier to write
            photometry = []
            for pkey in photkeys:
                for i in np.arange(len(self.photometry[pkey]['wl'])):
                    if 'err' in self.photometry[pkey].keys():
                        if 'ulim' in self.photometry[pkey].keys():
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, self.photometry[pkey]['ulim'][i]])
                        else:
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, None])
                    else:
                        if 'ulim' in self.photometry[pkey].keys():
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, self.photometry[pkey]['ulim'][i]])
                        else:
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, None])

            #Do the same for the spectra
            spectra = []
            for skey in speckeys:
                for i in np.arange(len(self.spectra[skey]['wl'])):
                    if 'err' in self.spectra[skey].keys():
                        spectra.append([self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], self.spectra[skey]['err'][i], skey])
                    else:
                        spectra.append([self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], np.nan, skey])

            photometry = np.array(photometry)
            spectra = np.array(spectra)

            #Now create the fits extensions using binary fits files
            if len(photkeys) != 0:
                photHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E', array=photometry[:,0]),\
                    fits.Column(name='lFl', format='E', array=photometry[:,1]),\
                    fits.Column(name='err', format='E', array=photometry[:,2]),\
                    fits.Column(name='instrument', format='A20', array=photometry[:,3]),\
                    fits.Column(name='ulim', format='E', array=photometry[:,4])])
            else:
                photHDU = fits.BinTableHDU.from_columns(
                    [fits.Column(name='wl', format='E'),\
                    fits.Column(name='lFl', format='E'),\
                    fits.Column(name='err', format='E'),\
                    fits.Column(name='instrument', format='A20'),\
                    fits.Column(name='ulim', format='E') ])

            if len(speckeys) != 0:
                specHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E', array=spectra[:,0]),\
                    fits.Column(name='lFl', format='E', array=spectra[:,1]),\
                    fits.Column(name='err', format='E', array=spectra[:,2]),\
                    fits.Column(name='instrument', format='A20', array=spectra[:,3])])
            else:
                specHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E'),\
                    fits.Column(name='lFl', format='E'),\
                    fits.Column(name='err', format='E'),\
                    fits.Column(name='instrument', format='A20')])

            #Denote which extension is which
            photHDU.header.set('FITS_EXT', 'PHOTOMETRY')
            specHDU.header.set('FITS_EXT', 'SPECTRA')

            #Build the primary header
            prihdr = fits.Header()
            prihdr['DERED'] = dered

            if Av:
                prihdr['AV'] = Av
            if extlaw:
                prihdr['EXTLAW'] = extlaw

            prihdr['COMMENT'] = 'This file contains photometry in extension 1'
            prihdr['COMMENT'] = 'and contains spectra in extension 2 in the'
            prihdr['COMMENT'] = 'form of binary fits tables. See '
            prihdr['COMMENT'] = 'http://docs.astropy.org/en/stable/io/fits/ for'
            prihdr['COMMENT'] = 'more information about binary fits tables.'
            prihdr['COMMENT'] = 'Extension 0 does not contain a table as primary'
            prihdr['COMMENT'] = 'FITS extensions cannot hold binary tables,'
            prihdr['COMMENT'] = 'but it does hold some information about the object'
            prihdr['COMMENT'] = 'in the header.'
            prihdr['COMMENT'] = '**********************************************'
            prihdr['COMMENT'] = 'This file was created with EDGE, which is'
            prihdr['COMMENT'] = 'here: https://github.com/cespaillat/EDGE .'

            priHDU = fits.PrimaryHDU(header=prihdr)

            HDU = fits.HDUList([priHDU, photHDU, specHDU])
            HDU.writeto(datapath+self.name+'_obs.fits', overwrite = overwrite)
        else:
            f = open(datapath+self.name+'_obs.csv', 'w')
            f.write('wavelength(microns),nuFnu(erg/s/cm2),error,instrument,upperlimit\n')
            for pkey in photkeys:
                for i in range(len(self.photometry[pkey]['wl'])):
                    if 'err' in self.photometry[pkey].keys():
                        if 'ulim' in self.photometry[pkey].keys():
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, self.photometry[pkey]['ulim'][i]))
                        else:
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, None))
                    else:
                        if 'ulim' in self.photometry[pkey].keys():
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, self.photometry[pkey]['ulim'][i]))
                        else:
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, None))
            for skey in speckeys:
                f = open(datapath+self.name+'_obs_spec'+skey+'.csv', 'w')
                f.write('wavelength(microns),nuFnu(erg/s/cm2),error,instrument\n')
                for i in np.arange(len(self.spectra[skey]['wl'])):
                    if 'err' in self.spectra[skey].keys():
                        f.write("{},{},{},{}\n".format(self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], self.spectra[skey]['err'][i], skey))
                    else:
                        f.write("{},{},{},{}\n".format(self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], np.nan, skey))

class Red_Obs(TTS_Obs):
    """
    A similar class to TTS_Obs, except meant to be utilized for observations that have not yet been
    dereddened. Once dereddened, the Red_Obs object will be saved as a TTS_Obs object fits file. If saved prior to
    dereddening, the fits file will be associated with Red_Obs instead. I recommend keeping both.

    """

    def dered(self, Av, law, datapath=datapath, Av_unc = 0, flux=1, lpath=commonpath, rv = None, err_prop=0, UV = 0, overwrite = False, \
        Mstar = None, Mref = None, Rstar = None, Rref = None, Tstar = None, Tref = None, dist = None, save = True, verbose = True):
        """
        Deredden the spectra/photometry present in the object, and then convert to TTS_Obs structure.
        This function is adapted from the IDL procedure 'dered_calc.pro' (written by Melissa McClure).
        This requires the spectral fluxes to be units of erg s-1 cm-2 cm-1.

        INPUTS
        Av: The Av extinction value.
        Av_unc: The uncertainty in the Av value provided.
        law: The extinction law to be used -- these extinction laws are found in the ext_laws.pkl file.
             The options you have are 'mkm09_rv5', 'mkm09_rv3', and 'mathis90_rv3.1', 'CCM'
        datapath: Where your dereddened observations fits will be saved.
        flux: BOOLEAN -- if True (1), the function will treat your photometry as flux units (erg s-1 cm-2).
              if False (0), the function will treat your photometry as being Flambda (erg s-1 cm-2 cm-1).
        lpath: Where the 'ext_laws.pkl' file is located. I suggest hard coding it as 'edgepath'.
        err_prop: BOOLEAN -- if True (1), will propagate the uncertainty of your photometry with the
                  uncertainty in your Av. Otherwise, it will not.
        UV: Uses dereddening law from Whittet et al. 2004 based on the extinction towards HD 29647
            for wavelengths between 0.125-9.33 microns.
            NOTE: This is ONLY useful for stars extincted by diffuse media, with RV = 3.1 (MATHIS LAW)
        overwrite: boolean, if set to True, will overwrite the old fits file

        OUTPUT
        Returns the dereddened object.
        Also creates a fits file called '[self.name]_obs.fits' in the path provided in datapath. If
        there is already an obs fits file there, it will add an integer to the name to differentiate
        between the two files, rather than overwriting.
        """

        # Read in the dereddening laws pickle. The default is whereever you keep EDGE.py, but you can move it.
        original = lpath + 'ext_laws.pkl'
        destination = "data_unix.pkl"
        content = ''
        with open(original, 'rb') as infile:
            content = infile.read()
        with open(destination, 'wb') as output:
            for line in content.splitlines():
                output.write(line + str.encode('\n'))
        with open(destination, 'rb') as infile:
            extLaws   = cPickle.load(infile, encoding = 'latin1')

        # Figure out which law we will be using based on the user input and Av:
        if law == 'mkm09_rv5':
            if verbose:
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
            if verbose:
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
            if verbose:
                print('Using the Mathis (1990) Rv=3.1 law\n(appropriate for diffuse ISM).')
            AjoAks   = 2.5341
            AvoAj    = 3.55                                     # for Rv=3.1
            wave_law = extLaws['mathis_rv3']['wl']
            ext_law  = extLaws['mathis_rv3']['ext']
        elif law == 'CCM':
            print('Using the CCM law.')
            AvoAj = 1.0
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
            if law == 'CCM':
                extInterpolated = util.reddccm89(self.spectra[specKey]['wl'], rv)
            else:
                extInterpolated = np.interp(self.spectra[specKey]['wl'], wave_law, ext_law) # Interpolated ext.

            #If the UV flag is on, replace the extinction law between 0.125 - .33 microns with a different law
            if UV == True:

                #Ensure that you are using the Rv = 3.1 Mathis law
                if law != 'mathis90_rv3.1' and law != 'mkm09_rv3':
                    raise ValueError('UV dereddening mode for use only with the low extinction laws (mathis90_rv3.1 and mkm09_rv3)')


                #Covert wavelength to 1/microns
                x = self.spectra[specKey]['wl'] ** (-1)

                #Define the valid range (3-8 (micron)^-1)
                UVrange = np.where((x > 1) & (x < 8))

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
                if err_prop:
                    spec_unc    = np.float64(spec_flux*np.sqrt(((self.spectra[specKey]['err']/self.spectra[specKey]['lFl'])**2.) + (((0.4*math.log(10)*extInterpolated*Av_unc)/(AvoAj))**2.)) )
                    print('Uncertainties in Av should not be treated as random uncertainties, they are systematic!')
                else:
                    spec_unc    = np.float64(self.spectra[specKey]['err']*10.0**(0.4*A_lambda))

            else:
                spec_unc    = None

            deredObs.add_spectra(specKey, self.spectra[specKey]['wl'], spec_flux, errors=spec_unc)


        # Spectra are done, onwards to photometry:
        for photKey in self.photometry.keys():
            if law == 'CCM':
                extInterpolated = util.reddccm89(self.photometry[photKey]['wl'], rv)
            else:
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
                    print('Uncertainties in Av should not be treated as random uncertainties, they are systematic!!!')
                    phot_err= np.float64(photcorr * np.sqrt(((errcorr/photcorr)**2.) + \
                                         (((0.4*math.log(10.)*extInterpolated*Av_unc)/AvoAj)**2.)) )
                else:
                    phot_err= np.float64(errcorr*10.0**(0.4*A_lambda)) # Without propogating error!
            else:
                phot_err    = None
            if 'ulim' in self.photometry[photKey].keys():
                ulimVal     = self.photometry[photKey]['ulim']
            else:
                ulimVal     = None
            # Now, convert everything to flux units:
            phot_dered      = phot_dered * self.photometry[photKey]['wl'] * 1e-4
            try:
                phot_err        = phot_err * self.photometry[photKey]['wl'] * 1e-4
            except TypeError:
                pass
            deredObs.add_photometry(photKey, self.photometry[photKey]['wl'], phot_dered, errors=phot_err, ulim=ulimVal, verbose = 0)

        # Now that the new TTS_Obs object has been created and filled in, we must save it:

        if save:
            deredObs.saveObs(datapath = datapath, overwrite = overwrite, Av = Av, extlaw = law)

        return deredObs



    def saveObs(self, datapath=datapath, overwrite = 1, make_csv = False, dered = False,\
        Av = None, extlaw = None, Mstar = None, Mref = None, Rstar = None, \
        Rref = None, Tstar = None, Tref = None, dist = None, dref = None):
        """
        Saves a TTS_Obs object as a fits file (Replacing pickle files)

        INPUTS:
            None

        OPTIONAL INPUTS:
            datapath:[string] Path where the data will be saved. Default is datapath
            overwrite:[boolean] Overwrites existing files if true
            Av: [float] Extinction (Av)
            extlaw: [string] Law used to de-redden data
            make_csv: [boolean] Makes two .csv files containing photometry and spectra if true
            dered: [boolean] True if the data has been dereddened. Default here is False, since this
                   is working with the Red_Obs class

            Mtar: [float] Mass of the star in Msun
            Mref: [string] Reference for Mstar

            Rstar: [float] Radius of the star in Rstar
            Rref: [string] Reference for Rstar

            Tstar: [float] Temperature of the star in K
            Tref: [string] Reference for Tstar

            dist: [float] Distance to the star in pc
            dref: [string] Reference for dist

        OUTPUT:
            A fits file with there extensions.
            The primary extension (extension 0) only has information in the header about the target
            The second extension (extension 1) contains photometry of the object in a binary fits table
            The third extension (extension 2) contains spectra of the object in a binary fits table

            Each table contains data points, errors, and the associated instrument. Extension 1 also contains
            if the object is an upper limit

        Author:
            Connor Robinson, October 19th, 2017
        """
        photkeys = list(self.photometry.keys())
        speckeys = list(self.spectra.keys())

        if make_csv == False:
            #Break the photometry down into something easier to write
            photometry = []
            for pkey in photkeys:
                for i in np.arange(len(self.photometry[pkey]['wl'])):
                    if 'err' in self.photometry[pkey].keys():
                        if 'ulim' in self.photometry[pkey].keys():
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, self.photometry[pkey]['ulim'][i]])
                        else:
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, None])
                    else:
                        if 'ulim' in self.photometry[pkey].keys():
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, self.photometry[pkey]['ulim'][i]])
                        else:
                            photometry.append([self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, None])

            #Do the same for the spectra
            spectra = []
            for skey in speckeys:
                for i in np.arange(len(self.spectra[skey]['wl'])):
                    if 'err' in self.spectra[skey].keys():
                        spectra.append([self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], self.spectra[skey]['err'][i], skey])
                    else:
                        spectra.append([self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], np.nan, skey])

            photometry = np.array(photometry)
            spectra = np.array(spectra)

            #Now create the fits extensions using binary fits files
            if len(photkeys) != 0:
                photHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E', array=photometry[:,0]),\
                    fits.Column(name='lFl', format='E', array=photometry[:,1]),\
                    fits.Column(name='err', format='E', array=photometry[:,2]),\
                    fits.Column(name='instrument', format='A20', array=photometry[:,3]),\
                    fits.Column(name='ulim', format='E', array=photometry[:,4])])
            else:
                photHDU = fits.BinTableHDU.from_columns(
                    [fits.Column(name='wl', format='E'),\
                    fits.Column(name='lFl', format='E'),\
                    fits.Column(name='err', format='E'),\
                    fits.Column(name='instrument', format='A20'),\
                    fits.Column(name='ulim', format='E') ])

            if len(speckeys) != 0:
                specHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E', array=spectra[:,0]),\
                    fits.Column(name='lFl', format='E', array=spectra[:,1]),\
                    fits.Column(name='err', format='E', array=spectra[:,2]),\
                    fits.Column(name='instrument', format='A20', array=spectra[:,3])])
            else:
                specHDU = fits.BinTableHDU.from_columns(\
                    [fits.Column(name='wl', format='E'),\
                    fits.Column(name='lFl', format='E'),\
                    fits.Column(name='err', format='E'),\
                    fits.Column(name='instrument', format='A20')])

            #Denote which extension is which
            photHDU.header.set('FITS_EXT', 'PHOTOMETRY')
            specHDU.header.set('FITS_EXT', 'SPECTRA')

            #Build the primary header
            prihdr = fits.Header()
            prihdr['DERED'] = dered

            if Av:
                prihdr['AV'] = Av
            if extlaw:
                prihdr['EXTLAW'] = extlaw

            prihdr['COMMENT'] = 'This file contains photometry in extension 1'
            prihdr['COMMENT'] = 'and contains spectra in extension 2 in the'
            prihdr['COMMENT'] = 'form of binary fits tables. See '
            prihdr['COMMENT'] = 'http://docs.astropy.org/en/stable/io/fits/ for'
            prihdr['COMMENT'] = 'more information about binary fits tables.'
            prihdr['COMMENT'] = 'Extension 0 does not contain a table as primary'
            prihdr['COMMENT'] = 'FITS extensions cannot hold binary tables,'
            prihdr['COMMENT'] = 'but it does hold some information about the object'
            prihdr['COMMENT'] = 'in the header.'
            prihdr['COMMENT'] = '**********************************************'
            prihdr['COMMENT'] = 'This file was created with EDGE, which is'
            prihdr['COMMENT'] = 'here: https://github.com/cespaillat/EDGE .'

            priHDU = fits.PrimaryHDU(header=prihdr)

            HDU = fits.HDUList([priHDU, photHDU, specHDU])
            HDU.writeto(datapath+self.name+'_red.fits', overwrite = overwrite)
        else:
            f = open(datapath+self.name+'_red.csv', 'w')
            f.write('wavelength(microns),nuFnu(erg/s/cm2),error,instrument,upperlimit\n')
            for pkey in photkeys:
                for i in range(len(self.photometry[pkey]['wl'])):
                    if 'err' in self.photometry[pkey].keys():
                        if 'ulim' in self.photometry[pkey].keys():
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, self.photometry[pkey]['ulim'][i]))
                        else:
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], self.photometry[pkey]['err'][i], pkey, None))
                    else:
                        if 'ulim' in self.photometry[pkey].keys():
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, self.photometry[pkey]['ulim'][i]))
                        else:
                            f.write("{},{},{},{},{}\n".format(self.photometry[pkey]['wl'][i], self.photometry[pkey]['lFl'][i], np.nan, pkey, None))
            for skey in speckeys:
                f = open(datapath+self.name+'_red_spec'+skey+'.csv', 'w')
                f.write('wavelength(microns),nuFnu(erg/s/cm2),error,instrument\n')
                for i in np.arange(len(self.spectra[skey]['wl'])):
                    if 'err' in self.spectra[skey].keys():
                        f.write("{},{},{},{}\n".format(self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], self.spectra[skey]['err'][i], skey))
                    else:
                        f.write("{},{},{},{}\n".format(self.spectra[skey]['wl'][i], self.spectra[skey]['lFl'][i], np.nan, skey))

    def SPPickle(self, picklepath, overwrite = False, fill = 3):
        """
        The new version of SPPickle, different so you can differentiate between red and dered pickles.

        INPUT
        picklepath: The path where the new pickle file will be located.
        overwrite: boolean, if set to True, will overwrite the old pickle
        fill: How many numbers used in the model files (4 = name_XXXX.fits).

        OUTPUT
        A new pickle file in picklepath, of the name [self.name]_red.pkl
        """

        # Check whether or not the pickle already exists:
        pathlist = glob(picklepath+'*')
        pathlist = [x[len(picklepath):] for x in pathlist]

        outname         = self.name + '_red.pkl'
        count           = 1
        while 1:
            if outname in pathlist and overwrite == False:
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
