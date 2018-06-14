#!/usr/bin/env python
from astropy.io import ascii
import numpy as np
import util

"""

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
    ZhexingLi (Dec 2017), Anneliese Rilinger (May/June 2018)
"""


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
filterwl = 0.36, 0.44, 0.55, 0.64, 0.79, 1.22, 1.63, 2.19, 3.45, 4.75, 3.6, 4.5, 5.8, 8.0, 23.7
zeropoint = 1.81e-20, 4.26e-20, 3.64e-20, 3.08e-20, 2.55e-20, 1.57e-20, 1.02e-20, \
            6.36e-21, 2.81e-21, 1.54e-21, 2.775e-21, 1.795e-21, 1.166e-21, 6.31e-22, 7.14e-23

# reads in magnitudes and calculates observed colors
input_mags = xu, xb, xv, xr, xi, xj, xh, xk, xl, xm, irac1, irac2, irac3, irac4, mips1
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


if table == 'kh95':             # read in Kenyon & Hartmann 1995 (KH95) table
    with open(commonpath+'color_tables/tabla_a5_kh95_rev', 'r') as tablekh95:
        next(tablekh95)
        for line in tablekh95:
            col = line.split()
            if col[0] == sptin:
                teff = float(col[2])        # selects KH95 colors for the input spectral type
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
                kminusw10 = float(col[10])
                kminusw20 = float(col[11])
                kminusw30 = float(col[12])
                kminusw40 = float(col[13])
                rminusi0c = vminusi0c - vminusr0c
                iminusj0c = vminusj0 - vminusi0c

# begins calculation of Av's
av1 = 1
# open text file to write outputs
f = open(outputfile, 'w')

# Av from V-R
wlmic_Rband = 0.64
if law == 'mathis':
    f.write('{}\n'.format('Using Mathis law'))
    if r == 5:
        f.write('{}\n'.format('Using Rv=5'))
        ar = util.reddrv5(av1, wlmic_Rband, commonpath)
    else:
        f.write('{}\n'.format('Using Rv=3.1'))
        ar = util.redd(av1, wlmic_Rband, commonpath)
if law == 'HD29647':
    f.write('{}\n'.format('Using HD29647 law'))
    ar = util.reddhd(av1, wlmic_Rband, commonpath)
if law == 'CCM89':
    f.write('{}\n'.format('Using CCM89 law'))
    ar = util.reddccm89(wlmic_Rband, r, commonpath)
if law == 'mcclure':
    f.write('{}\n'.format('Using McClure (2010) law'))
    if r == 5:
        if avin>8:
            f.write('{}\n'.format('with mcclure_avgt8_rv5p0'))
            ar = util.reddmcclureavgt8rv5p0(av1, wlmic_Rband, commonpath)
        else:
            f.write('{}\n'.format('with mcclure_avlt8_rv5p0'))
            ar = util.reddmcclureavlt8rv5p0(av1, wlmic_Rband, commonpath)
    else:
        if avin>8:
            f.write('{}\n'.format('with mcclure_avgt8_rv3p1'))
            ar = util.reddmcclureavgt8rv3p1(av1, wlmic_Rband, commonpath)
        else:
            f.write('{}\n'.format('with mcclure_avlt8_rv3p1'))
            ar = util.reddmcclureavlt8rv3p1(av1, wlmic_Rband, commonpath)
avvminusrc=(1/(1.-ar))*(vminusr-vminusr0c)
f.write('{}\t{:.2}\n'.format('Av(V-R)=', avvminusrc))

# Av from V-Ic
wlmic_Iband = 0.79
if law == 'mathis':
    if r == 5:
        ai = util.reddrv5(av1, wlmic_Iband, commonpath)
    else:
        ai = util.redd(av1, wlmic_Iband, commonpath)
if law == 'HD29647':
    ai=util.reddhd(av1, wlmic_Iband, commonpath)
if law == 'CCM89':
    ai = util.reddccm89(wlmic_Iband, r, commonpath)
if law == 'mcclure':
    if r == 5:
        if avin > 8:
            ai = util.reddmcclureavgt8rv5p0(av1, wlmic_Iband, commonpath)
        else:
            ai = util.reddmcclureavlt8rv5p0(av1, wlmic_Iband, commonpath)
    else:
        if avin > 8:
            ai = util.reddmcclureavgt8rv3p1(av1, wlmic_Iband, commonpath)
        else:
            ai = util.reddmcclureavlt8rv3p1(av1, wlmic_Iband, commonpath)
avvminusic = (1/(1.-ai))*(vminusi-vminusi0c)
f.write('{}\t{:.2}\n'.format('Av(V-I)=', avvminusic))

# Av from Rc-Ic
avrcminusic = (1/(ar-ai))*(rminusi-rminusi0c)
f.write('{}\t{:.2}\n'.format('Av(R-I)=', avrcminusic))

# Av from Ic-J
wlmic_Jband = 1.22
if law == 'mathis':
    if r == 5:
        aj = util.reddrv5(av1, wlmic_Jband, commonpath)
    else:
        aj = util.redd(av1, wlmic_Jband, commonpath)
if law == 'HD29647':
    aj = util.reddhd(av1, wlmic_Jband, commonpath)
if law == 'CCM89':
    aj = util.reddccm89(wlmic_Jband, r, commonpath)
if law == 'mcclure':
    if r == 5:
        if avin > 8:
            aj = util.reddmcclureavgt8rv5p0(av1, wlmic_Jband, commonpath)
        else:
            aj = util.reddmcclureavlt8rv5p0(av1, wlmic_Jband, commonpath)
    else:
        if avin > 8:
            aj = util.reddmcclureavgt8rv3p1(av1, wlmic_Jband, commonpath)
        else:
            aj = util.reddmcclureavlt8rv3p1(av1, wlmic_Jband, commonpath)
avicminusj = (1/(ai-aj))*(iminusj-iminusj0c)
f.write('{}\t{:.2}\n'.format('Av(I-J)=', avicminusj))

# Adopt Av from V-I
# f.write('{}\n'.format('Observed Magnitudes:'))
# f.write('{}\n'.format(input_mags))

if inter == 'True':
    print('\n')
    print('User input Av is:', avin, '. To stick with it, press "y".', '\n')
    print('Calculated Av(V-R) is:', str(avvminusrc), '. To use it, press "1".', '\n')
    print('Calculated Av(V-I) is:', str(avvminusic), '. To use it, press "2".', '\n')
    print('Calculated Av(I-J) is:', str(avicminusj), '. To use it, press "3".', '\n')
    print('Calculated Av(R-I) is:', str(avrcminusic), '. To use it, press "4".', '\n')

    Av_new = input()
    if Av_new == 'y':
        Av_new = avin
    elif Av_new == '1':
        Av_new = avvminusrc
    elif Av_new == '2':
        Av_new = avvminusic
    elif Av_new == '3':
        Av_new = avicminusj
    elif Av_new == '4':
        Av_new = avrcminusic
else:
    Av_new = avin


f.write('{}\n'.format('----------------------------'))
f.write('{}\n'.format('Band Wavelength Mag_obs Mag_dered lFl_obs lFl_dered'))
# correct for reddening and calculate Flambda and Fobserved
for i in range(0, 15):
    filter = filterwl[i]
    # de-reddened magnitudes
    redmag = input_mags[i]
    if Av_new < 8:
        if r == 5:
            dered = util.reddrv5(av1, filter, commonpath)*Av_new
        else:
            dered = util.redd(av1, filter, commonpath)*Av_new
    else:
        if r == 5:
            dered = util.reddmcclureavgt8rv5p0(av1, filter, commonpath)*Av_new
        else:
            dered = util.reddmcclureavgt8rv3p1(av1, filter, commonpath)*Av_new
    dered_mag = input_mags[i]-dered
    # store de-reddened V, I, and J-band
    if i == 2:
        dered_V = input_mags[i]-dered
    if i == 4:
        dered_Ic = input_mags[i]-dered
    if i == 5:
        dered_J = input_mags[i]-dered

    # f.write('{}\t{:.3}\t{:.3}\n'.format(band[i], redmag, dered_mags))
    fnu = zeropoint[i]*10.**(-dered_mag/2.5)  # dereddened flux
    fnuobs = zeropoint[i]*10.**(-redmag/2.5)  # observed flux
    nu = c*1e4/filterwl[i]  # frequency corresponding to wavelength
    wangs = filterwl[i]*1.e4   # wavelength in angstroms
    nufnu = np.log10(nu*fnu)
    nufnuobs = np.log10(nu*fnuobs)
    fl = nu*fnu/wangs
    # store flux in U band for use in Lacc calculation
    if i == 0:
        flux = fl
    f.write('{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n'.format(band[i], filterwl[i], redmag, dered_mag, nufnuobs, nufnu))

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
if teff < 4169:
    valuesbaraffe = util.parameters_baraffe(teff,luminosity,commonpath)
    massbaraffe = valuesbaraffe[0]
    agebaraffe = valuesbaraffe[1]
else:
    massbaraffe = 99.0
    agebaraffe = 99.e6

f.write('{}\n'.format('----------------------------'))
f.write('{}\t{:.2}\t{:.2}\n'.format('M_Baraffe,Age_Baraffe', massbaraffe, agebaraffe/1.e6))

# Siess tracks
valuessiess = util.parameters_siess(teff, luminosity, commonpath)
masssiess = valuessiess[0]
agesiess = valuessiess[1]
f.write('{}\t{:.2}\t{:.2}\n'.format('M_Siess,Age_Siess', masssiess, agesiess/1.e6))

# Adopt user picked values
if isochrone == 'baraffe':
    mass = massbaraffe
    age = agebaraffe
else:
    mass = masssiess
    age = agesiess

f.write('{}\t{:.2}\t{:.2}\n'.format('M_adopt,Age_adopt', mass, age/1.e6))

# check out if switches around 598

# Mass accretion rate calculation
deltau = 680
lumu = 4*np.pi*(distance*pc)*(distance*pc/lsun)*flux*deltau

# f.write('{}\t{:.2}\t{:.2}\n'.format('flux, lumu', flux, lumu))
# U standard from I (ie, assuming no veiling at U)

# f.write('{:.2}\t{:.2}\t{:.2}\n'.format(uminusv0[0], vmi0c[0], dered_Ic))
ustandard = (uminusv0+vminusi0c)+dered_Ic

# f.write('{}\t{:.3}\n'.format('ustandard', ustandard[0]))
fnustandard = zeropoint[0]*10.**(-ustandard/2.5)
nu = c*1e4/filterwl[0]
wangs = filterwl[0]*1.e4
# flambda per A
fluxstandard = nu*fnustandard/wangs
lumustandard = 4.*np.pi*(distance*pc)*(distance*pc/lsun)*fluxstandard*deltau

# f.write('{}\t{:.2}\t{:.2}\n'.format('fluxstandard, lumustandard', fluxstandard[0], lumustandard[0]))

# excess U luminosity
lumu = lumu-lumustandard

# f.write('{}\t{:.2}\n'.format('lumu', lumu[0]))

if lumu > 0:
    # accretion luminosity - Gullbring calibration
    lacc = 1.09*np.log10(lumu)+0.98
    lacc = 10.**lacc
    if mass < 50:
        mdot = radius*lacc/gg/mass*(rsun/msun)*(lsun/msun)*3.17e7
    else:
        mdot = 0.
else:
    lacc = 0.
    mdot = 0.

# Plot HR diagram with object on it
if HR == 'True':
    figure = util.HRdiagram(teff, luminosity, isochrone, obj, commonpath)
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
if calcphot == 'yes' and table == 'kh95':

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
            flux_phot = util.interp(wl_phot[i], wl_from_colors, fluxes_from_colors, 12, i0)
            flux_phot = 10**flux_phot
        else:
            flux_phot = (10**fluxes_from_colors[11])*((wl_from_colors[11]/wl_phot[i])**3)
        filephot.write('{}\t{}\n'.format(wl_phot[i], flux_phot))
    filephot.close()

# calculate scaled template photosphere using PM13 colors
if calcphot == 'yes' and table == 'pm13':

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
            flux_phot = util.interp(wl_phot[i], wl_from_colors, fluxes_from_colors, 12, i0)
            flux_phot = 10**flux_phot
        else:
            flux_phot = (10**fluxes_from_colors[7])*((wl_from_colors[7]/wl_phot[i])**3)
        filephot.write('{}\t{}\n'.format(wl_phot[i], flux_phot))
    filephot.close()
