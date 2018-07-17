#!/usr/bin/env python
from astropy.io import ascii
import numpy as np
import util
import scipy.interpolate as sinterp
import os
import matplotlib.pyplot as plt
import math
import pickle
import scipy.optimize as op

starparampath = os.path.dirname(os.path.realpath(__file__))+'/'
commonpath = starparampath+'../COMMON/'

#-------------------------------------------------------------
def parameters_isochrone(tstar, lui, isomodel='siess', commonpath=commonpath):
    """
    Calculates age and mass using Siess & Forestini 1996 tracks or
    Baraffe et al. 1998 tracks.

    INPUTS
        tstar: stellar temperature
        lui:  stellar luminosity
        isomodel: Isochrones to be used, either 'siess' or 'baraffe'
    OUTPUT
        mass: stellar mass
        age: stellar age

    """
    lstar = np.log10(lui)
    tlstar = np.log10(tstar)

    if np.isnan(tlstar) or np.isnan(lstar):
        print('WARNING: Nans in temperature and/or luminosity. '+
        'Unable to calculate mass and age. \n')
        return np.nan, np.nan

    # reads in tracks and corresponding ages
    if isomodel == 'siess':
        siesspath = commonpath+'isochrones/siess/'
        fileiso = [siesspath+'siess_t_0.0005.dat', \
                   siesspath+'siess_t_0.0006.dat', \
                   siesspath+'siess_t_0.0007.dat', \
                   siesspath+'siess_t_0.0008.dat', \
                   siesspath+'siess_t_0.0009.dat', \
                   siesspath+'siess_t_0.001.dat', \
                   siesspath+'siess_t_0.002.dat', \
                   siesspath+'siess_t_0.003.dat', \
                   siesspath+'siess_t_0.004.dat', \
                   siesspath+'siess_t_0.005.dat', \
                   siesspath+'siess_t_0.006.dat', \
                   siesspath+'siess_t_0.007.dat', \
                   siesspath+'siess_t_0.008.dat', \
                   siesspath+'siess_t_0.009.dat', \
                   siesspath+'siess_t_0.01.dat', \
                   siesspath+'siess_t_0.02.dat', \
                   siesspath+'siess_t_0.03.dat', \
                   siesspath+'siess_t_0.04.dat', \
                   siesspath+'siess_t_0.05.dat', \
                   siesspath+'siess_t_1.dat', \
                   siesspath+'siess_t_10.dat']
        ages = [5.e5,6.e5,7.e5,8.e5,9.e5,1.e6,2.e6,3.e6,4.e6,5.e6,6.e6,\
        7.e6,8.e6,9.e6,1.e7,2.e7,3.e7,4.e7,5.e7,1.e9,1.e10]
        cols = [0,1,2] # columns of interest of files
    elif isomodel == 'baraffe':
        baraffepath = commonpath+'isochrones/baraffe/'
        fileiso = [baraffepath+'bhac15_t_0.0005.dat', \
                    baraffepath+'bhac15_t_0.001.dat', \
                    baraffepath+'bhac15_t_0.002.dat', \
                    baraffepath+'bhac15_t_0.003.dat', \
                    baraffepath+'bhac15_t_0.004.dat', \
                    baraffepath+'bhac15_t_0.005.dat', \
                    baraffepath+'bhac15_t_0.008.dat',\
                    baraffepath+'bhac15_t_0.01.dat',\
                    baraffepath+'bhac15_t_0.015.dat', \
                    baraffepath+'bhac15_t_0.02.dat', \
                    baraffepath+'bhac15_t_0.025.dat', \
                    baraffepath+'bhac15_t_0.03.dat', \
                    baraffepath+'bhac15_t_0.04.dat', \
                    baraffepath+'bhac15_t_0.05.dat',\
                    baraffepath+'bhac15_t_1.dat',\
                    baraffepath+'bhac15_t_10.dat']
        ages = [5.e5,1.e6,2.e6,3.e6,4.e6,5.e6,8.e6,1.e7,1.5e7,2.e7,2.5e7,3.e7,4.e7,5.e7,1.e9,1.e10]
        cols = [1,2,4] # columns of interest of files
    elif (isomodel == 'MIST') or (isomodel == 'mist'):
        f = open(commonpath+'isochrones/MIST_evol_tracks/MIST_tracks.pkl','rb')
        tracks = pickle.load(f)
        f.close()
        points = tracks[:,2:]
        values_mass = tracks[:,0]
        values_age = tracks[:,1]
    else:
        print('WARNING: '+isomodel+"is not supported. Use 'siess', 'baraffe', "+
        "or 'MIST'")
        return np.nan, np.nan

    if (isomodel == 'siess') or (isomodel == 'baraffe'):
        nages = len(ages)
        # We read the isochrones and build a grid of points
        # in the (Tstar,Lstar) plane
        points = []
        values_mass = []
        values_age = []
        for i in range(0,nages):
            tableiso = np.genfromtxt(fileiso[i],skip_header=1,
            usecols=(cols[0],cols[1],cols[2]))
            massint = tableiso[:,0]
            nmasses = len(massint)
            tint = np.log10(tableiso[:,1]).reshape(nmasses,1)
            lint = tableiso[:,2].reshape(nmasses,1)  # already in log
            points.append(np.concatenate((tint,lint),axis=1))
            values_mass.append(massint.reshape(nmasses))
            values_age.append(np.ones((nmasses))*ages[i])
        points = np.concatenate(points) # (Teff, Luminosities)
        values_mass = np.concatenate(values_mass)
        values_age = np.concatenate(values_age)

    # if out of isochrone range, reject
    if (tlstar > np.max(points[:,0])) or (tlstar < np.min(points[:,0])):
        print('WARNING: Temperature out of isochrone bounds')
        return np.nan, np.nan
    if (lstar > np.max(points[:,1])) or (lstar < np.min(points[:,1])):
        print('WARNING: Luminosity out of isochrone bounds')
        return np.nan, np.nan

    # Interpolation in mass and age
    mass = sinterp.griddata(points, values_mass, (np.array([tlstar,lstar])), method='linear')
    age = sinterp.griddata(points, values_age, (np.array([tlstar,lstar])), method='linear')

    if np.isnan(mass) or np.isnan(age):
        print('WARNING: Problem with interpolation. Values are probably '+
        'out of bounds')
    return mass[0], age[0]

#-------------------------------------------------------------
def reddening(wl, law, rv=3.1, av=1, avin=0, commonpath=commonpath):
    '''
    Calculates reddening correction using specified law.

    INPUTS:
    - wl: wavelength.
    - law: extinction law that will be used.
    - av: visual extinction.

    OUTPUT:
    - red: reddening correction.
    '''
    if law == 'mathis':
        redden = redd(av, wl, rv, commonpath)
    elif law == 'HD29647':
        redden = reddhd(av, wl, commonpath)
    elif law == 'CCM89':
        redden = reddccm89(wl, r, commonpath)
    elif law == 'mcclure':
        if avin > 3.0:
            redden = reddmcclure(av, wl, rv, avin, commonpath)
        else:
            redden = redd(av, wl, rv, commonpath)
    else:
        raise IOError('Unknown extinction law.')

    return redden

#-------------------------------------------------------------
def redd(av, wl, rv=3.1, commonpath=commonpath):
    """
    Calculates reddening correction using Mathis ARAA 28,37,1990

    INPUTS
        av: visual extinction
        wl: wavelength

    OUTPUT
        redd: reddening correction

    """
    tablemathis=ascii.read(commonpath+'ext_laws/'+'mathis.table.rev', delimiter=" ")
    wlm=tablemathis['wlmu']
    if rv == 3.1:
        al=tablemathis['A_wl/A_J_R3.1']
    elif rv == 5:
        al=tablemathis['A_wl/A_J_R5']

    # normalize to V
    avv=al[wlm == 0.55]
    al=al/avv

    redd=np.interp(wl,wlm[::-1],al[::-1])*av

    return redd

#-------------------------------------------------------------
def reddmcclure(av, wl, rv=3.1, avin=0, commonpath=commonpath):
    """
    Calculates reddening correction using McClure (2010)

    INPUTS
        av: visual extinction
        wl: wavelength
        rv: Rv.
        avin: initial (maybe guess) extinction
    OUTPUT
        redd: reddening correction
    """
    if rv == 5:
        if avin>8:
            table=ascii.read(commonpath+'ext_laws/'+
            'mcclurereddening_avgt8_rv5p0', data_start=1)
        else:
            table=ascii.read(commonpath+'ext_laws/'+
            'mcclurereddening_avlt8_rv5p0', data_start=1)
    elif rv ==3.1:
        if avin>8:
            table=ascii.read(commonpath+'ext_laws/'+
            'mcclurereddening_avgt8_rv3p1', data_start=1)
        else:
            table=ascii.read(commonpath+'ext_laws/'+
            'mcclurereddening_avlt8_rv3p1', data_start=1)

    wlm=table['col1']
    al=table['col2']

    redd=np.interp(wl,wlm,al)*av

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
    table=ascii.read(commonpath+'ext_laws/'+'hd29647_ext_pei_1.dat', delimiter=" ")
    wlm=table['col1']
    al=table['col2']
    wlm=wlm*1e-4

    redd=np.interp(wl,wlm[::-1],al[::-1])*av

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
def HRdiagram(tstar,lui,track,obj,commonpath):
    """
    Plots object of interest on a HR diagram with selected isochrones.

    INPUTS
        tstar: temperature of the star
        lui: luminostiy of the star
        obj: object name

    OUTPUT
        HR diagram plot
    """

    # Plotting parameters
    params = {
            'font.family': 'serif',
            'mathtext.fontset': 'cm',
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'axes.labelsize': 13,
            'font.size': 8,
            'legend.fontsize': 14,
            'xtick.labelsize': 15,
            'ytick.labelsize': 15,
            'text.usetex': False,
            'figure.figsize': [6, 4]
            }
    plt.rcParams.update(params)

    #plot_isochrones = True
    # Required paths (photometry, spectra, and figures)
    path_data = commonpath + 'isochrones/'
#    path_figures = outpath

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot regions
    plot_test, = plt.plot(3850,0.6,'o', ms=6, mfc='red', mec='None', label=obj)

    # create legend for objects
    objects_legend = plt.legend(handles=[plot_test], loc=3,
                                numpoints=1, frameon=False,
                                handletextpad=0.01, markerfirst=True)
    # Add the legend manually to the current Axes.
    ax1 = plt.gca().add_artist(objects_legend)

    # plot isochrones if requested
    if track == 'baraffe':               #plot HR using baraffe tracks
        isochrone_handles = []
        ages = [1, 3, 5,10]
        styles = {1:'-', 3: '-.', 5:'--',10: ':'}
        for age in ages:
            # read isochrones from Baraffe+15
            isochrone = ascii.read('{}bhac15_t_{}.dat'.format(path_data+'baraffe/',age*1e-3))
            i_isochrones = isochrone['teff'].data >2000
            teffs_iso = isochrone['teff'].data[i_isochrones]
            lums_iso = 10**isochrone['logL'].data[i_isochrones]
            plot_isochrone, = plt.plot(teffs_iso, lums_iso,
                                       label='{} Myr'.format(age), ls=styles[age], color='k', lw=2,)
            isochrone_handles.append(plot_isochrone)
        # create legend for isochrones
        isochrone_legend = plt.legend(handles=isochrone_handles, loc=1,
                                      frameon=False, markerfirst=False)
    else:                               #plot HR using siess tracks
        isochrone_handles = []
        ages = [1, 3, 5,10]
        styles = {1:'-', 3: '-.', 5:'--',10: ':'}
        for age in ages:
            # read isochrones from Siess
            isochrone = ascii.read('{}siess_t_{}.dat'.format(path_data+'siess/',age*1e-3))
            i_isochrones = isochrone['teff'].data >2000
            teffs_iso = isochrone['teff'].data[i_isochrones]
            lums_iso = 10**isochrone['logL'].data[i_isochrones]
            plot_isochrone, = plt.plot(teffs_iso, lums_iso,
                                       label='{} Myr'.format(age), ls=styles[age], color='k', lw=2,)
            isochrone_handles.append(plot_isochrone)
        # create legend for isochrones
        isochrone_legend = plt.legend(handles=isochrone_handles, loc=1,
                                      frameon=False, markerfirst=False)


    plt.xlabel(r'$\mathrm{T_{eff}\,[K]}$',size=20)
    plt.ylabel(r'$\mathrm{L_*\,[L_{\odot}]}$',size=20)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1.2e4, 1.7e3)
    plt.ylim(3.01e-4, 4e2)
    #plt.show()

    # Format axis
    ax.set_yticks([1e-3, 1e-1, 1e1, 1e3])
    #tickFormatter = FormatStrFormatter('%d')
    #ax.xaxis.set_major_formatter( tickFormatter )
    #ax.xaxis.set_minor_formatter( tickFormatter )
    ax.set_xticks([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, ], minor='True')
    ax.set_xticklabels(['2000', '3000', '4000', '', '6000', '', '', ''], minor='True')
    #plt.setp(ax.get_xminorticklabels(), visible=False)

    # lets try changing the
    #plt.savefig('{}HR_diagram_Baraffe2015.pdf'.format(path_figures),dpi=300,bbox_inches='tight')
    #plt.close()

    return fig

def color_tables(sptin, table='kh95', commonpath=commonpath):
    '''
    '''
    if table == 'kh95':      # read in Kenyon & Hartmann 1995 (KH95) table
        with open(commonpath+'color_tables/tabla_a5_kh95_rev','r') as tablekh95:
            next(tablekh95)
            for line in tablekh95:
                col = line.split()
                if col[0] == sptin: # selects KH95 colors for the input spectral type
                    teff = float(col[2])
                    bolcorr = float(col[3])
                    uminusv0 = float(col[4])
                    bminusv0 = float(col[5])
                    uminusb0 = uminusv0 - bminusv0
                    vminusr0c = float(col[6])
                    vminusr0j = float(col[7])
                    vminusi0c = float(col[8])
                    vminusi0j = float(col[9])
                    vminusj0 = float(col[10])
                    vminush0 = float(col[11])
                    vminusk0 = float(col[12])
                    vminusl0 = float(col[13])
                    vminusm0 = float(col[14])
                    jminush0 = vminush0 - vminusj0
                    rminusi0c = vminusi0c - vminusr0c
                    iminusj0c = vminusj0-vminusi0c
                    break
        colors0 = np.array([uminusb0, bminusv0, vminusr0c, rminusi0c,
        iminusj0c, jminush0, vminusk0-vminush0])
        return teff, colors0, bolcorr, uminusv0, bminusv0, uminusb0, vminusr0c,\
        vminusr0j, vminusi0c, vminusi0j, vminusj0, vminush0, vminusk0, vminusl0,\
        vminusm0, jminush0, rminusi0c, iminusj0c

    else:   # read in Pecaut & Mamajek 2013 (PM13) table
        with open(commonpath+'color_tables/tabla_no5_pm13', 'r') as tablepm13:
            # some columns missing in this table, missing values are replaced by 0
            for i in range(25):
                next(tablepm13)
            for line in tablepm13:
                col = line.split()
                if col[0] == sptin + 'V':
                    teff = float(col[1])
                    bolcorr = float(col[2])
                    uminusb0 = float(col[3])
                    bminusv0 = float(col[4])
                    uminusv0 = uminusb0 + bminusv0
                    vminusr0c = float(col[5])
                    vminusi0c = float(col[6])
                    vminusj0 = float(col[7])
                    vminush0 = float(col[8])
                    vminusk0 = float(col[9])
                    jminush0 = vminush0 - vminusj0
                    rminusi0c = vminusi0c - vminusr0c
                    iminusj0c = vminusj0 - vminusi0c
                    try: # Some sp. types do not have these colors
                        kminusw10 = float(col[10])
                        kminusw20 = float(col[11])
                        kminusw30 = float(col[12])
                        kminusw40 = float(col[13])
                    except:
                        kminusw10 = np.nan
                        kminusw20 = np.nan
                        kminusw30 = np.nan
                        kminusw40 = np.nan
                    break
        colors0 = np.array([uminusb0, bminusv0, vminusr0c, rminusi0c, iminusj0c,
        jminush0, vminusk0-vminush0])
        return teff, colors0, bolcorr, uminusb0, bminusv0, uminusv0, vminusr0c,\
        vminusi0c, vminusj0, vminush0, vminusk0, jminush0, kminusw10, kminusw20,\
        kminusw30, kminusw40, rminusi0c, iminusj0c

# def photoshpere(Jmag, colors0=None, sptin=None, table='kh95', commonpath=commonpath):
#     if colors0:
#         continue
#     elif sptin:
#         print('Not working yet')
#     Hmag = Jmag - colors0[5]
#     Kmag = Hmag - colors0[6]
#     Imag = Jmag + colors0[4]
#     Rmag = Imag + colors0[3]
#     Vmag = Rmag + colors0[2]
#     Bmag = Vmag + colors0[1]
#     Umag = Bmag + colors0[0]

def deredenned_magnitudes(av, mags, wls, rv=3.1, law='mcclure',
commonpath=commonpath):
    '''
    '''
    dered_mags = []
    for i in range(len(mags)):
        # de-reddened magnitudes
        dered_mags.append(mags[i] - reddening(wls[i], law=law, rv=rv, avin=av,
        commonpath=commonpath) * av)
    dered_mags = np.array(dered_mags)
    return dered_mags

def residual_extinctions_J(pos, mags, wls, colors0, rv=3.1, law='mcclure',
commonpath=commonpath):
    '''
    '''
    av = pos[0]
    Jmag = pos[1]

    dered_mags = deredenned_magnitudes(av, mags, wls, rv, law, commonpath)

    # Colors from table
    Hmag = Jmag - colors0[5]
    Kmag = Hmag - colors0[6]
    Imag = Jmag + colors0[4]
    Rmag = Imag + colors0[3]
    Vmag = Rmag + colors0[2]
    # Bmag = Vmag + colors0[1]
    # Umag = Bmag + colors0[0]
    table_mags = np.array([Vmag,Rmag,Imag,Jmag,Hmag,Kmag])

    table_mags = table_mags[np.isnan(dered_mags) == False]
    dered_mags = dered_mags[np.isnan(dered_mags) == False]
    return (table_mags - dered_mags)

def residual_extinctions(pos, mags, wls, colors0, rv=3.1, law='mcclure',
commonpath=commonpath):
    '''
    '''
    av = pos[0]

    dered_mags = deredenned_magnitudes(av, mags, wls, rv, law, commonpath)
    Jmag = dered_mags[5]

    # Colors from table
    Hmag = Jmag - colors0[5]
    Kmag = Hmag - colors0[6]
    Imag = Jmag + colors0[4]
    Rmag = Imag + colors0[3]
    Vmag = Rmag + colors0[2]
    # Bmag = Vmag + colors0[1]
    # Umag = Bmag + colors0[0]
    table_mags = np.array([Vmag,Rmag,Imag,Jmag,Hmag,Kmag])

    table_mags = table_mags[np.isnan(dered_mags) == False]
    dered_mags = dered_mags[np.isnan(dered_mags) == False]
    return (table_mags - dered_mags)

#-------------------------------------------------------------
def starparam(obj, sptin, avin, distance, law='mcclure', table='kh95',
isochrone='MIST', HR=True, calcphot=True, inter=True, r=3.1,
xu=np.nan, xb=np.nan, xv=np.nan, xr=np.nan, xi=np.nan, xj=np.nan, xh=np.nan,
xk=np.nan, xl=np.nan, xm=np.nan, irac1=np.nan, irac2=np.nan, irac3=np.nan,
irac4=np.nan, mips1=np.nan, outpath='./',
outputfile='starparam.txt', photfile='photosphere.txt', commonpath=commonpath,
photfilewl=commonpath+'wavelengths/'+'longitudes_4testruns_shorter.ent'):
    '''
    starparam.py

    PURPOSE:
        reads in SpT and photometry (in magnitudes, UBVRcIcJHK,IRAC,MIPS)
        uses SpT to get Teff, colors, and BC from either KH95 or PM13
        dereddens magnitudes using input Av
        calculates Av using V-R,V-I,R-I,I-J
        can use extinction laws from Mathis, Cardelli et al., McClure
        calculates L from dereddened J-band and KH95 formula
        calculates M and age interpolating in Baraffe and Siess tracks
        calculates Mdot from U-band using Gullbring et al 1998 calibration
        can also get template photosphere, assumes Rayleigh-Jeans for long wavelengths

    NOTES:
        run with jobstarparam.py

    Edited by:
        ZhexingLi (Dec 2017), Anneliese Rilinger (May/June 2018),
        Enrique Macias (June 2018)
    '''

    # constants
    c = 2.99793e10
    pc = 3.08e18
    gg = 6.67e-8
    msun = 1.989e33
    rsun = 6.9599e10
    lsun = 3.826e33

    # stores bands, wavelengths, and zeropoints (erg/cm2/s/Hz) from
    # Bessell 1979, Johson66(Jsystem),Bessell&Brett 1988, Spitzer webpage
    band = 'U', 'B', 'V', 'Rc', 'Ic', 'J', 'H', 'K', 'L', 'M', '[3.6]', '[4.5]', '[5.8]', '[8]', '[24]'
    filterwl = [0.36, 0.44, 0.55, 0.64, 0.79, 1.22, 1.63, 2.19, 3.45, 4.75, 3.6, 4.5, 5.8, 8.0, 23.7]
    zeropoint = 1.81e-20, 4.26e-20, 3.64e-20, 3.08e-20, 2.55e-20, 1.57e-20, 1.02e-20, \
                6.36e-21, 2.81e-21, 1.54e-21, 2.775e-21, 1.795e-21, 1.166e-21, 6.31e-22, 7.14e-23

    # reads in magnitudes and calculates observed colors
    input_mags = [xu, xb, xv, xr, xi, xj, xh, xk, xl, xm, irac1, irac2, irac3, irac4, mips1]
    v = xv
    uminusb = xu-xb
    bminusv = xb-v
    vminusr = xv-xr
    rminusi = xr-xi
    vminusi = vminusr+rminusi
    iminusj = xi-xj
    jminush = xj-xh
    hminusk = xh-xk
    kminusl = xk-xl

    if table == 'kh95':   # read in Kenyon & Hartmann 1995 (KH95) table
        teff, colors0, bolcorr, uminusv0, bminusv0, uminusb0, vminusr0c,\
        vminusr0j, vminusi0c, vminusi0j, vminusj0, vminush0, vminusk0, vminusl0,\
        vminusm0, jminush0, rminusi0c, iminusj0c = color_tables(sptin, table, commonpath)

    else:   # read in Pecaut & Mamajek 2013 (PM13) table
        teff, colors0, bolcorr, uminusb0, bminusv0, uminusv0, vminusr0c,\
        vminusi0c, vminusj0, vminush0, vminusk0, jminush0, kminusw10, kminusw20,\
        kminusw30, kminusw40, rminusi0c, iminusj0c = color_tables(sptin, table, commonpath)

    # open text file to write outputs
    f = open(outputfile, 'w')
    # begins calculation of Av's
    if law == 'mathis':
        f.write('{}\n'.format('Using Mathis law'))
        if r == 5:
            f.write('{}\n'.format('Using Rv=5'))
        else:
            f.write('{}\n'.format('Using Rv=3.1'))
    elif law == 'HD29647':
        f.write('{}\n'.format('Using HD29647 law'))
    elif law == 'CCM89':
        f.write('{}\n'.format('Using CCM89 law'))
    elif law == 'mcclure':
        f.write('{}\n'.format('Using McClure (2010) law'))
        if avin < 3:
            f.write('{}\n'.format("AV<3, so McClure's law uses Mathis"))
            if r == 5:
                f.write('{}\n'.format('Using Rv=5'))
            else:
                f.write('{}\n'.format('Using Rv=3.1'))
        else:
            if r == 5:
                if avin>8:
                    f.write('{}\n'.format('with mcclure_avgt8_rv5p0'))
                else:
                    f.write('{}\n'.format('with mcclure_avlt8_rv5p0'))
            else:
                if avin>8:
                    f.write('{}\n'.format('with mcclure_avgt8_rv3p1'))
                else:
                    f.write('{}\n'.format('with mcclure_avlt8_rv3p1'))

    try:
        # Av from fit to input magnitudes, from V to K
        init = [avin]
        result, flag = op.leastsq(residual_extinctions, init,
        args=(input_mags[2:8], filterwl[2:8], colors0, r, law, commonpath))
        avfit1 = result[0]
        if flag != 5:
            f.write('{}\t{:.2}\n'.format('Av(fit)=', avfit1))
        else:
            f.write('{}\t{:.2}\n'.format('Av(fit)=', np.nan))
    except:
        print('AV_Fit1 did not work. You probably need more data points.')
        avfit1 = np.nan
        f.write('{}\t{:.2}\n'.format('Av(fit)=', avfit1))

    try:
        # Fitting for a dereddened J as well
        init = [avin, xj]
        result, flag = op.leastsq(residual_extinctions_J, init,
        args=(input_mags[2:8], filterwl[2:8], colors0, r, law, commonpath))
        avfit2 = result[0]
        Jfit = result[1]
        if flag != 5:
            f.write('{}\t{:.2}, {:.2}\n'.format('Av(fit), J =', avfit2, Jfit))
        else:
            f.write('{}\t{:.2}, {:.2}\n'.format('Av(fit), J =', np.nan, np.nan))
    except:
        print('AV_Fit2 did not work. You probably need more data points.')
        avfit2 = np.nan
        Jfit = np.nan
        f.write('{}\t{:.2}, {:.2}\n'.format('Av(fit), J =', avfit2, Jfit))

    # Av from V-R
    wlmic_Rband = 0.64
    ar_av = reddening(wlmic_Rband, law=law, rv=r, avin=avin, commonpath=commonpath)
    avvminusrc=(1./(1.-ar_av))*(vminusr-vminusr0c)
    f.write('{}\t{:.2}\n'.format('Av(V-R)=', avvminusrc))

    # Av from V-Ic
    wlmic_Iband = 0.79
    ai_av = reddening(wlmic_Iband, law=law, rv=r, avin=avin, commonpath=commonpath)
    avvminusic = (1./(1.-ai_av))*(vminusi-vminusi0c)
    f.write('{}\t{:.2}\n'.format('Av(V-I)=', avvminusic))

    # Av from Rc-Ic
    avrcminusic = (1./(ar_av-ai_av))*(rminusi-rminusi0c)
    f.write('{}\t{:.2}\n'.format('Av(R-I)=', avrcminusic))

    # Av from Ic-J
    wlmic_Jband = 1.22
    aj_av = reddening(wlmic_Jband, law=law, rv=r, avin=avin, commonpath=commonpath)
    avicminusj = (1/(ai_av-aj_av))*(iminusj-iminusj0c)
    f.write('{}\t{:.2}\n'.format('Av(I-J)=', avicminusj))

    if inter:
        print('\n')
        print('User input Av is:', avin, '. To stick with it, press "y".', '\n')
        print('Calculated Av(fit1) is:', str(avfit1), '. To use it, press "1".', '\n')
        print('Calculated Av(fit2) is:', str(avfit2), '. To use it, together with the obtained dereddened J. press "2".', '\n')
        print('Calculated Av(V-R) is:', str(avvminusrc), '. To use it, press "3".', '\n')
        print('Calculated Av(V-I) is:', str(avvminusic), '. To use it, press "4".', '\n')
        print('Calculated Av(I-J) is:', str(avicminusj), '. To use it, press "5".', '\n')
        print('Calculated Av(R-I) is:', str(avrcminusic), '. To use it, press "6".', '\n')

        Av_choice = input()
        if Av_choice == 'y':
            Av_new = avin
        elif Av_choice == '1':
            Av_new = avfit1
        elif Av_choice == '2':
            Av_new = avfit2
            dered_J = Jfit
            dered_V = dered_J + vminusj0
            dered_Ic = dered_V - vminusi0c
        elif Av_choice == '3':
            Av_new = avvminusrc
        elif Av_choice == '4':
            Av_new = avvminusic
        elif Av_choice == '5':
            Av_new = avicminusj
        elif Av_choice == '6':
            Av_new = avrcminusic
    else:
        Av_new = avin

    f.write('{}\n'.format('----------------------------'))
    f.write('{}\n'.format('Band Wavelength Mag_obs Mag_dered lFl_obs lFl_dered'))
    # correct for reddening and calculate Flambda and Fobserved
    for i in range(len(filterwl)):
        filter = filterwl[i]
        # de-reddened magnitudes
        redmag = input_mags[i]
        dered = reddening(filter, law=law, rv=r, avin=Av_new,
        commonpath=commonpath) * Av_new
        dered_mag = input_mags[i]-dered
        if Av_choice != '2':
            # store de-reddened V, I, and J-band
            if i == 2:
                dered_V = dered_mag
            if i == 4:
                dered_Ic = dered_mag
            if i == 5:
                dered_J = dered_mag

        fnu = zeropoint[i] * 10.**(-dered_mag/2.5)  # dereddened flux
        fnuobs = zeropoint[i] * 10.**(-redmag/2.5)  # observed flux
        nu = c * 1e4 / filterwl[i]  # frequency corresponding to wavelength
        nufnu = np.log10(nu * fnu)
        nufnuobs = np.log10(nu * fnuobs)
        # store flux in U band for use in Lacc calculation
        if i == 0:
            wangs = filterwl[i] * 1.e4   # wavelength in angstroms
            flux = nu * fnu / wangs  # store flambda in U band for use in Lacc calculation
        f.write('{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n'.format(
        band[i], filterwl[i], redmag, dered_mag, nufnuobs, nufnu))

    dmodulus = 5*np.log10(distance)-5   # distance modulus = m - M
    abs_jmag = dered_J-dmodulus
    vmkstand = vminusk0		# standard V - K from color table
    mbol = abs_jmag+0.10*vmkstand+1.17   # bolometric magnitude?
    mbolv = bolcorr+(dered_V-dmodulus)   # how is this different than mbol?
    lum = (4.75-mbol)/2.5
    lum = 10.**lum
    lumv = (4.75-mbolv)/2.5
    lumv = 10.**lumv

    # f.write('{}\n'.format('dmodulus,abs_jmag,vmkstand,mbol,lum'))
    # f.write('{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\n'.format(dmodulus,abs_jmag,vmkstand[0],mbol[0],lum[0]))
    # f.write('{}\t{:.2}\t{:.2}\n'.format('L(J), L(V)',lum[0], lumv[0]))

    # for Teff < G5 in Table 5 of KH95
    # adopts L with Mbol from J following KH95
    # for higher from V
    if table == 'kh95':
        if teff < 5770: # temp of KH95 G5
            luminosity = lum
            radius = np.sqrt(luminosity)/(teff/5770.)**2
            radiusv = np.sqrt(lumv)/(teff/5770.)**2
        else:
            luminosity = lumv
            radiusv = np.sqrt(lumv)/(teff/5770.)**2
            radius = radiusv
    else:
        if teff < 5660: # temp of PM13 G5
            luminosity = lum
            radius = np.sqrt(luminosity)/(teff/5770.)**2
            radiusv = np.sqrt(lumv)/(teff/5770.)**2
        else:
            luminosity = lumv
            radiusv = np.sqrt(lumv)/(teff/5770.)**2
            radius = radiusv
    # print 'R(J), R(V)', radius[0], radiusv[0]
    # f.write('{}\t{:.2}\t{:.2}\n'.format('R(J), R(V)', radius[0], radiusv[0]))

    # calculate mass and age with both Siess and Baraffe tracks
    # Baraffe tracks
    #if teff < 4169:
    valuesbaraffe = parameters_isochrone(teff, luminosity, 'baraffe', commonpath)
    massbaraffe = valuesbaraffe[0]
    agebaraffe = valuesbaraffe[1]

    f.write('{}\n'.format('----------------------------'))
    f.write('{}\t{:.2}\t{:.2}\n'.format('M_Baraffe,Age_Baraffe', massbaraffe, agebaraffe/1.e6))

    # Siess tracks
    valuessiess = parameters_isochrone(teff, luminosity, 'siess', commonpath)
    masssiess = valuessiess[0]
    agesiess = valuessiess[1]
    f.write('{}\t{:.2}\t{:.2}\n'.format('M_Siess,Age_Siess', masssiess, agesiess/1.e6))

    # MIST tracks
    valuesmist = parameters_isochrone(teff, luminosity, 'MIST', commonpath)
    massmist = valuesmist[0]
    agemist = valuesmist[1]
    f.write('{}\t{:.2}\t{:.2}\n'.format('M_MIST,Age_MIST', massmist, agemist/1.e6))

    # Adopt user picked values
    if isochrone == 'baraffe':
        mass = massbaraffe
        age = agebaraffe
    elif isochrone == 'siess':
        mass = masssiess
        age = agesiess
    elif (isochrone == 'MIST') or (isochrone == 'mist'):
        mass = massmist
        age = agemist

    f.write('{}\t{:.2}\t{:.2}\n'.format('M_adopt,Age_adopt', mass, age/1.e6))

    # Mass accretion rate calculation
    deltau = 680
    lumu = 4*np.pi*(distance*pc)**2./lsun*flux*deltau

    # U standard from I (ie, assuming no veiling at U)
    ustandard = (uminusv0+vminusi0c)+dered_Ic
    fnustandard = zeropoint[0] * 10.**(-ustandard/2.5)
    nu = c * 1e4/filterwl[0]
    wangs = filterwl[0]*1.e4
    # flambda per A
    fluxstandard = nu * fnustandard/wangs
    lumustandard = 4.*np.pi*(distance*pc)**2./lsun*fluxstandard*deltau

    # excess U luminosity
    lumu = lumu-lumustandard
    if lumu > 0:
        # accretion luminosity - Gullbring calibration
        lacc = 1.09*np.log10(lumu)+0.98
        lacc = 10.**lacc
        mdot = radius*lacc/gg/mass*(rsun/msun)*(lsun/msun)*3.17e7
    else:
        lacc = 0.
        mdot = 0.

    # Plot HR diagram with object on it
    if HR:
        figure = HRdiagram(teff, luminosity, isochrone, obj, commonpath)
        figure.show()
        figure.savefig(outpath + 'HR_' + obj + '.png', dpi=300, bbox_inches='tight')
    else:
        pass

    f.write('{}\n'.format('----------------------------'))
    f.write('{}\t{}\n'.format('Spectral Type', sptin))
    f.write('{}\t{}\n'.format('Teff (K)', teff))
    f.write('{}\t{:.2}\n'.format('L (Lsun)', luminosity))
    f.write('{}\t{:.2}\n'.format('R (Rsun)', radius))
    f.write('{}\t{:.2}\n'.format('M (Msun)', mass))
    f.write('{}\t{:.2}\n'.format('Age (Myr)', age/1.e6))
    f.write('{}\t{:.2}\n'.format('Lacc (Lsun)', lacc))
    f.write('{}\t{:.2}\n'.format('Mdot (Msun/yr)', mdot))

    f.close()


    #####################################################

    # calculate scaled template photosphere using KH95 colors
    if calcphot and table == 'kh95':

        i0 = 0

        filephotwl = ascii.read(photfilewl, Reader=ascii.NoHeader, data_start=0, data_end=1)
        number_wl = filephotwl['col1'][0]  # number of wavelengths to calculate photosphere fluxes for
        filephotwl = ascii.read(photfilewl, Reader=ascii.NoHeader, data_start=1)
        wl_phot = filephotwl['col1']  # list of wavelengths to calculate photosphere fluxes for

        filephot = open(photfile, 'w')

        vmag = vminusj0 + dered_J
        bmag = bminusv0 + vmag
        umag = uminusb0 + bmag
        r0cmag = vmag - vminusr0c
        r0jmag = vmag - vminusr0j
        i0cmag = vmag - vminusi0c
        i0jmag = vmag - vminusi0j
        hmag = vmag - vminush0
        kmag = vmag - vminusk0
        lmag = vmag - vminusl0
        mmag = vmag - vminusm0

        mags_from_colors = [umag, bmag, vmag, r0cmag, r0jmag, i0cmag, i0jmag, dered_J, hmag, kmag, lmag, mmag]
        wl_from_colors = [0.36, 0.44, 0.55, 0.64, 0.7, 0.79, 0.9, 1.22, 1.63, 2.19, 3.45, 4.75]
        zeropoint_from_colors = [1.81e-20, 4.26e-20, 3.64e-20, 3.08e-20, 3.01e-20, 2.55e-20,
                                 2.43e-20, 1.57e-20, 1.02e-20, 6.36e-21, 2.81e-21, 1.54e-21]

        fluxes_from_colors = []
        for i in range(0, 12):
            fnu_from_colors = zeropoint_from_colors[i]*10.**(-mags_from_colors[i]/2.5)
            nu_from_colors = c*1e4/wl_from_colors[i]
            wangs_from_colors = wl_from_colors[i]*1.e4
            xfl = nu_from_colors*fnu_from_colors/wangs_from_colors
            log_nu_fnu_from_colors = np.log10(nu_from_colors*fnu_from_colors)
            fluxes_from_colors.append(log_nu_fnu_from_colors)
        for i in range(0, number_wl):
            if wl_phot[i] < wl_from_colors[11]:
                flux_phot = np.interp(wl_phot[i], wl_from_colors, fluxes_from_colors)
                flux_phot = 10**flux_phot
            else:
                flux_phot = (10**fluxes_from_colors[11])*((wl_from_colors[11]/wl_phot[i])**3)
            filephot.write('{}\t{}\n'.format(wl_phot[i], flux_phot))
        filephot.close()

    # calculate scaled template photosphere using PM13 colors
    if calcphot and table == 'pm13':

        i0 = 0

        filephotwl = ascii.read(photfilewl, Reader=ascii.NoHeader, data_start=0, data_end=1)
        number_wl = filephotwl['col1'][0]
        filephotwl = ascii.read(photfilewl, Reader=ascii.NoHeader, data_start=1)
        wl_phot = filephotwl['col1']

        filephot = open(photfile, 'w')

        vmag = vminusj0+dered_J
        bmag = bminusv0+vmag
        umag = uminusb0+bmag
        r0cmag = vmag-vminusr0c
        i0cmag = vmag-vminusi0c
        hmag = vmag-vminush0
        kmag = vmag-vminusk0
        w1mag = kmag-kminusw10
        w2mag = kmag-kminusw20
        w3mag = kmag-kminusw30
        w4mag = kmag-kminusw40

        mags_from_colors = [umag, bmag, vmag, r0cmag, i0cmag, dered_J, hmag, kmag, w1mag, w2mag, w3mag, w4mag]
        wl_from_colors = [0.36, 0.44, 0.55, 0.64, 0.79, 1.22, 1.63, 2.19, 3.35, 4.60, 11.56, 22.09]
        # zero points - erg/cm2/s/Hz
        # from Bessell 1979,BB88 nir, Jarrett et al. 2011 WISE
        # for Jarrett convert Fnu (Jy) using 1Jy=1e-23 erg/s/cm2/Hz
        zeropoint_from_colors = [1.81e-20, 4.26e-20, 3.64e-20, 3.08e-20, 2.55e-20, 1.57e-20, 1.02e-20,
                                 6.36e-21, 3.10e-21, 1.72e-21, 3.17e-22, 8.36e-23]

        fluxes_from_colors = []
        for i in range(0, 12):
            fnu_from_colors = zeropoint_from_colors[i]*10.**(-mags_from_colors[i]/2.5)
            nu_from_colors = c*1e4/wl_from_colors[i]   # frequency in Hertz
            wangs_from_colors = wl_from_colors[i]*1.e4  # wavelength in Angstroms
            xfl = nu_from_colors*fnu_from_colors/wangs_from_colors
            log_nu_fnu_from_colors = np.log10(nu_from_colors*fnu_from_colors)
            fluxes_from_colors.append(log_nu_fnu_from_colors)
        for i in range(0, number_wl):
            if wl_phot[i] < wl_from_colors[7]:
                flux_phot = np.interp(wl_phot[i], wl_from_colors, fluxes_from_colors)
                flux_phot = 10**flux_phot
            else:
                flux_phot = (10**fluxes_from_colors[7])*((wl_from_colors[7]/wl_phot[i])**3)
            filephot.write('{}\t{}\n'.format(wl_phot[i], flux_phot))
        filephot.close()
