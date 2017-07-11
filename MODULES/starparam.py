#!/usr/bin/env python

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
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

"""


# constants
c=2.99793e10
pc=3.08e18
gg=6.67e-8
msun=1.989e33
rsun=6.9599e10
lsun=3.826e33

# stores bands, wavelengths, and zeropoints (erg/cm2/s/Hz) from 
# Bessell 1979, Johson66(Jsystem),Bessell&Brett 1988, Spitzer webpage 
# 0:U,1:b,2:V,3:Rc,4:Ic,5:J,6:H,7:K,8:L,9:M,
# 10:irac[3.6],11:irac[4.5],12:irac[5.8],13:irac[8],14:mips[24]
band='U','B','V','Rc','Ic','J','H','K','L','M','[3.6]','[4.5]','[5.8]','[8]','[24]'
wef=0.36,0.44,0.55,0.64,0.79,1.22,1.63,2.19,3.45,4.75,3.6,4.5,5.8,8.0,23.7
zp=1.81e-20,4.26e-20,3.64e-20,3.08e-20,2.55e-20,1.57e-20,1.02e-20,6.36e-21,2.81e-21,1.54e-21,2.775e-21,1.795e-21,1.166e-21,6.31e-22,7.14e-23

# reads in magnitudes and calculates observed colors
pro=xu,xb,xv,xr,xi,xj,xh,xk,xl,xm,irac1,irac2,irac3,irac4,mips1
v=xv
umb=xu-xb
bmv=xb-v
vmr=xv-xr
rmi=xr-xi
vmi=vmr+rmi
imj=xi-xj
jmh=xj-xh
hmk=xh-xk
kml=xk-xl

if table == 'kh95':

# read in Kenyon & Hartmann 1995 (KH95) table
	tablekh95=ascii.read(commonpath+'color_tables/'+'tabla_a5_kh95_rev', delimiter=" ")
	sptkh = tablekh95['SpT']
	tefkh = tablekh95['Tef']
	bckh = tablekh95['BC']
	umvkh = tablekh95['U-V']
	bmvkh = tablekh95['B-V']
	vmrckh = tablekh95['V-Rc']
	vmrjkh = tablekh95['V-Rj']
	vmickh = tablekh95['V-Ic']
	vmijkh = tablekh95['V-Ij']
	vmjkh = tablekh95['V-J']
	vmhkh = tablekh95['V-H']
	vmkkh = tablekh95['V-K']
	vmlkh = tablekh95['V-L']
	vmmkh = tablekh95['V-M']

# selects KH95 colors for the input spectral type
	umv0=umvkh[sptkh == sptin]
	bmv0=bmvkh[sptkh == sptin]
	umb0=umv0-bmv0
	vmr0c=vmrckh[sptkh == sptin]
	vmi0c=vmickh[sptkh == sptin]
	vmr0j=vmrjkh[sptkh == sptin]
	vmi0j=vmijkh[sptkh == sptin]
	vmj0=vmjkh[sptkh == sptin]
	vmh0=vmhkh[sptkh == sptin]
	vmk0=vmkkh[sptkh == sptin]
	vml0=vmlkh[sptkh == sptin]
	vmm0=vmmkh[sptkh == sptin]
	jmh0=vmh0-vmj0
	rmi0c=vmickh[sptkh == sptin]-vmrckh[sptkh == sptin]
	imj0c=vmjkh[sptkh == sptin]-vmickh[sptkh == sptin]
	teff=tefkh[sptkh == sptin]
	bc=bckh[sptkh == sptin]
	
else:

# read in Pecaut & Mamajek 2013 (PM13) table
	tablepm13=ascii.read(commonpath+'color_tables/'+'tabla_no5_pm13')
	sptpm = tablepm13['SpType']
	tefpm = tablepm13['Teff']
	bcpm = tablepm13['BCV']
	umbpm = tablepm13['U-B']
	bmvpm = tablepm13['B-V']
	vmrcpm = tablepm13['V-Rc']
	vmicpm = tablepm13['V-Ic']
	vmjpm = tablepm13['V-J']
	vmhpm = tablepm13['V-H']
	vmkpm = tablepm13['V-Ks']
	kmw1pm = tablepm13['K-W1']
	kmw2pm = tablepm13['K-W2']
	kmw3pm = tablepm13['K-W3']
	kmw4pm = tablepm13['K-W4']

# selects PM13 colors for the input spectral type
	sptin=sptin+'V'
	umb0=umbpm[sptpm == sptin]
	bmv0=bmvpm[sptpm == sptin]
	umv0=umb0+bmv0
	vmr0c=vmrcpm[sptpm == sptin]
	vmi0c=vmicpm[sptpm == sptin]
	vmj0=vmjpm[sptpm == sptin]
	vmh0=vmhpm[sptpm == sptin]
	vmk0=vmkpm[sptpm == sptin]
	jmh0=vmh0-vmj0
	rmi0c=vmicpm[sptpm == sptin]-vmrcpm[sptpm == sptin]
	imj0c=vmjpm[sptpm == sptin]-vmicpm[sptpm == sptin]
	kmw10=kmw1pm[sptpm == sptin]
	kmw20=kmw2pm[sptpm == sptin]
	kmw30=kmw3pm[sptpm == sptin]
	kmw40=kmw4pm[sptpm == sptin]
	teff=tefpm[sptpm == sptin]
	bc=bcpm[sptpm == sptin]

# begins calculation of Av's
av1=1

#open text file instead of writing to screen
f = open(outputfile, 'w')
#f.write('{}\t{:.2}\n'.format('Av =', avin))

# Av from V-R
wlmic_Rband=0.64 
if law == 'mathis':
	#print 'Using Mathis law'
	f.write('{}\n'.format('Using Mathis law'))
	if r == 5:
		#print 'Using Rv=5'
		f.write('{}\n'.format('Using Rv=5'))
		ar=util.reddrv5(av1,wlmic_Rband,commonpath)
	else:
		#print 'Using Rv=3.1'
		f.write('{}\n'.format('Using Rv=3.1'))
		ar=util.redd(av1,wlmic_Rband,commonpath)
if law == 'HD29647':
	#print 'Using HD29647 law'
	f.write('{}\n'.format('Using HD29647 law'))
	ar=util.reddhd(av1,wlmic_Rband,commonpath)
if law == 'CCM89':
	#print 'Using CCM89 law'
	f.write('{}\n'.format('Using CCM89 law'))
	ar=util.reddccm89(wlmic_Rband,r,commonpath)
if law == 'mcclure':
	#print 'Using McClure (2010) law'
	f.write('{}\n'.format('Using McClure (2010) law'))
	if r == 5:
		if avin>8:
			#print 'with mcclure_avgt8_rv5p0'
			f.write('{}\n'.format('with mcclure_avgt8_rv5p0'))
			ar=util.reddmcclureavgt8rv5p0(av1,wlmic_Rband,commonpath)
		else:
			#print 'with mcclure_avlt8_rv5p0'
			f.write('{}\n'.format('with mcclure_avlt8_rv5p0'))
			ar=util.reddmcclureavlt8rv5p0(av1,wlmic_Rband,commonpath)			
	else:
		if avin>8:
			#print 'with mcclure_avgt8_rv3p1'
			f.write('{}\n'.format('with mcclure_avgt8_rv3p1'))
			ar=util.reddmcclureavgt8rv3p1(av1,wlmic_Rband,commonpath)
		else:
			#print 'with mcclure_avlt8_rv3p1'
			f.write('{}\n'.format('with mcclure_avlt8_rv3p1'))
			ar=util.reddmcclureavlt8rv3p1(av1,wlmic_Rband,commonpath)	
avvmrc=(1/(1.-ar))*(vmr-vmr0c)
#print 'Av(V-R)', avvmrc[0]
f.write('{}\t{:.2}\n'.format('Av(V-R)=', avvmrc[0]))

# Av from V-Ic
wlmic_Iband=0.79
if law == 'mathis':
	if r == 5:
		ai=util.reddrv5(av1,wlmic_Iband,commonpath)
	else:
		ai=util.redd(av1,wlmic_Iband,commonpath)
if law == 'HD29647':
	ai=util.reddhd(av1,wlmic_Iband,commonpath)
if law == 'CCM89':
	ai=util.reddccm89(wlmic_Iband,r,commonpath)
if law == 'mcclure':
	if r == 5:
		if avin>8:
			ai=util.reddmcclureavgt8rv5p0(av1,wlmic_Iband,commonpath)
		else:
			ai=util.reddmcclureavlt8rv5p0(av1,wlmic_Iband,commonpath)			
	else:
		if avin>8:
			ai=util.reddmcclureavgt8rv3p1(av1,wlmic_Iband,commonpath)
		else:
			ai=util.reddmcclureavlt8rv3p1(av1,wlmic_Iband,commonpath)	
avvmic=(1/(1.-ai))*(vmi-vmi0c)
#print 'Av(V-I)', avvmic[0]
f.write('{}\t{:.2}\n'.format('Av(V-I)=', avvmic[0]))

# Av from Rc-Ic
avrcmic=(1/(ar-ai))*(rmi-rmi0c)
#print 'Av(R-I)', avrcmic[0]
f.write('{}\t{:.2}\n'.format('Av(R-I)=', avvmic[0]))

# Av from Ic-J
wlmic_Jband=1.22
if law == 'mathis':
	if r == 5:
		aj=util.reddrv5(av1,wlmic_Jband,commonpath)
	else:
		aj=util.redd(av1,wlmic_Jband,commonpath)
if law == 'HD29647':
	aj=util.reddhd(av1,wlmic_Jband,commonpath)		
if law == 'CCM89':
	aj=util.reddccm89(wlmic_Jband,r,commonpath)
if law == 'mcclure':
	if r == 5:
		if avin>8:
			aj=util.reddmcclureavgt8rv5p0(av1,wlmic_Jband,commonpath)
		else:
			aj=util.reddmcclureavlt8rv5p0(av1,wlmic_Jband,commonpath)			
	else:
		if avin>8:
			aj=util.reddmcclureavgt8rv3p1(av1,wlmic_Jband,commonpath)
		else:
			aj=util.reddmcclureavlt8rv3p1(av1,wlmic_Jband,commonpath)	
avicmj=(1/(ai-aj))*(imj-imj0c)
#print 'Av(I-J)',avicmj[0]
f.write('{}\t{:.2}\n'.format('Av(I-J)=', avicmj[0]))

# Adopt Av from V-I
#print 'Observed Magnitudes:'
#print pro
#f.write('{}\n'.format('Observed Magnitudes:'))
#f.write('{}\n'.format(pro))

f.write('{}\n'.format('----------------------------'))
f.write('{}\n'.format('Band Wavelength Mag_obs Mag_dered lFl_obs lFl_dered'))
# correct for reddening and calculate Flambda and Fobserved
for i in range(0,15):
	wlmic=wef[i]
	# de-reddened magnitudes
	redmag=pro[i]
	if law == 'mathis':
		if r == 5:
			dered=util.reddrv5(av1,wlmic,commonpath)*avin
		else:
			dered=util.redd(av1,wlmic,commonpath)*avin
	if law == 'HD29647':
		dered=util.reddhd(av1,wlmic,commonpath)*avin
	if law == 'CCM89':
		dered=util.reddccm89(wlmic,r,commonpath)*avin
	if law == 'mcclure':
		if r == 5:
			if avin>8:
				dered=util.reddmcclureavgt8rv5p0(av1,wlmic,commonpath)*avin
			else:
				dered=util.reddmcclureavlt8rv5p0(av1,wlmic,commonpath)*avin		
		else:
			if avin>8:
				dered=util.reddmcclureavgt8rv3p1(av1,wlmic,commonpath)*avin
			else:
				dered=util.reddmcclureavlt8rv3p1(av1,wlmic,commonpath)*avin
	dpro=pro[i]-dered	
	# store de-reddened V, I, and J-band
	if i==2:
		dproV=pro[i]-dered		
	if i==4:
		dproIc=pro[i]-dered
	if i==5:
		dproJ=pro[i]-dered
			
	#print band[i], redmag, dpro
	#f.write('{}\t{:.3}\t{:.3}\n'.format(band[i], redmag, dpro))
	fnu=zp[i]*10.**(-dpro/2.5)
	fnuobs=zp[i]*10.**(-redmag/2.5)
	nu=c*1e4/wef[i]
	wangs=wef[i]*1.e4
	nufnu=np.log10(nu*fnu)
	nufnuobs=np.log10(nu*fnuobs)
	fl=nu*fnu/wangs
	# store V-band derived flux
	if i==0: 
		flux=fl
	#print wef[i], nufnuobs,nufnu
	f.write('{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\n'.format(band[i], wef[i], redmag, dpro, nufnuobs,nufnu))

dmodulus=5*np.log10(distance)-5
xmj=dproJ-dmodulus
vmkstand=vmk0
mbol=xmj+0.10*vmkstand+1.17
mbolv=bc+(dproV-dmodulus)
lum=(4.75-mbol)/2.5
lum=10.**lum
lumv=(4.75-mbolv)/2.5
lumv=10.**lumv	
#print 'dmodulus,xmj,vmkstand,mbol,lum'
#print dmodulus,xmj,vmkstand[0],mbol[0],lum[0]
#print 'L(J), L(V)',lum[0], lumv[0]
#f.write('{}\n'.format('dmodulus,xmj,vmkstand,mbol,lum'))
#f.write('{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\n'.format(dmodulus,xmj,vmkstand[0],mbol[0],lum[0]))
#f.write('{}\t{:.2}\t{:.2}\n'.format('L(J), L(V)',lum[0], lumv[0]))

# for Teff < G5 in Table 5 of KH95 
# adopts L with Mbol from J following KH95
# for higher from V
if table=='kh95': 
	if teff < 5770: #temp of KH95 G5
		luminosity=lum[0]
		radius=np.sqrt(luminosity)/(teff/5770.)**2
		radiusv=np.sqrt(lumv)/(teff/5770.)**2
	else:
		luminosity=lumv[0]
		radiusv=np.sqrt(lumv)/(teff/5770.)**2
		radius=radiusv
else:
	if teff < 5660: #temp of PM13 G5
		luminosity=lum[0]
		radius=np.sqrt(luminosity)/(teff/5770.)**2
		radiusv=np.sqrt(lumv)/(teff/5770.)**2
	else:
		luminosity=lumv[0]
		radiusv=np.sqrt(lumv)/(teff/5770.)**2
		radius=radiusv
#print 'R(J), R(V)', radius[0], radiusv[0]
#f.write('{}\t{:.2}\t{:.2}\n'.format('R(J), R(V)', radius[0], radiusv[0]))

# calculate mass and age with both Siess and Baraffe tracks
if teff<4169:
# Baraffe tracks
	valuesbaraffe=util.parameters_baraffe(teff,luminosity,commonpath)
	massbaraffe=valuesbaraffe[0]
	agebaraffe=valuesbaraffe[1]
else:
	massbaraffe=99.0
	agebaraffe=99.e6
#print 'M_Baraffe,Age_Baraffe',massbaraffe,agebaraffe/1.e6
f.write('{}\n'.format('----------------------------'))
f.write('{}\t{:.2}\t{:.2}\n'.format('M_Baraffe,Age_Baraffe',massbaraffe,agebaraffe/1.e6))

# Siess tracks
valuessiess=util.parameters_siess(teff[0],luminosity,commonpath)
masssiess=valuessiess[0]
agesiess=valuessiess[1]
#print 'M_Siess,Age_Siess',masssiess,agesiess/1.e6
f.write('{}\t{:.2}\t{:.2}\n'.format('M_Siess,Age_Siess',masssiess,agesiess/1.e6))

# Adopt Siess values
mass=masssiess
age=agesiess
#print 'M_adopt,Age_adopt',mass,age/1.e6
f.write('{}\t{:.2}\t{:.2}\n'.format('M_adopt,Age_adopt',mass,age/1.e6))

#check out if switches around 598

# Mass accretion rate calculation
deltau=680
lumu=4*np.pi*(distance*pc)*(distance*pc/lsun)*flux*deltau
#print 'flux,lumu',flux,lumu
#f.write('{}\t{:.2}\t{:.2}\n'.format('flux,lumu',flux,lumu))
# U standard from I (ie, assuming no veiling at U)
#print umv0[0],vmi0c[0],dproIc
#f.write('{:.2}\t{:.2}\t{:.2}\n'.format(umv0[0],vmi0c[0],dproIc))
ustandard=(umv0+vmi0c)+dproIc
#print 'ustandard',ustandard[0]
#f.write('{}\t{:.3}\n'.format('ustandard',ustandard[0]))
fnustandard=zp[0]*10.**(-ustandard/2.5)
nu=c*1e4/wef[0]
wangs=wef[0]*1.e4
# flambda per A
fluxstandard=nu*fnustandard/wangs
lumustandard=4.*np.pi*(distance*pc)*(distance*pc/lsun)*fluxstandard*deltau
#print 'fluxstandard,lumustandard',fluxstandard[0],lumustandard[0]
#f.write('{}\t{:.2}\t{:.2}\n'.format('fluxstandard,lumustandard',fluxstandard[0],lumustandard[0]))
# excess U luminosity
lumu=lumu-lumustandard
#print 'lumu',lumu[0]
#f.write('{}\t{:.2}\n'.format('lumu',lumu[0]))

if lumu>0:
# accretion luminosity - Gullbring calibration
	lacc=1.09*np.log10(lumu)+0.98
	lacc=10.**lacc
	if mass<50:
		mdot=radius*lacc/gg/mass*(rsun/msun)*(lsun/msun)*3.17e7
	else:
		mdot=[]
		mdot.append(0.)
else:
	lacc=[]
	lacc.append(0.)
	mdot=[]
	mdot.append(0.)

#print 'Teff L Rs Ms <Av> Age(myr),Lacc,Mdot'
#print teff[0],luminosity[0],radius[0],mass,avin,age/1.e6,lacc[0],mdot[0]
f.write('{}\n'.format('----------------------------'))
f.write('{}\t{}\n'.format('Spectral Type', sptin))
f.write('{}\t{}\n'.format('Teff (K)', teff[0]))
f.write('{}\t{:.2}\n'.format('L (Lsun)', luminosity))
f.write('{}\t{:.2}\n'.format('R (Rsun)', radius[0]))
f.write('{}\t{:.2}\n'.format('M (Msun)', mass))
f.write('{}\t{:.2}\n'.format('Age (Myr)', age/1.e6))
f.write('{}\t{:.2}\n'.format('Lacc (Lsun)', lacc[0]))
f.write('{}\t{:.2}\n'.format('Mdot (Msun/yr)', mdot[0]))

f.close()


#####################################################

# calculate scaled template photosphere using KH95 colors
if calcphot=='yes' and table=='kh95':

	i0=0

	filephotwl=ascii.read(photfilewl,Reader=ascii.NoHeader,data_start=0,data_end=1)
	nwl=filephotwl['col1'][0]
	filephotwl=ascii.read(photfilewl,Reader=ascii.NoHeader,data_start=1)
	wl_phot=filephotwl['col1']

	filephot = open(photfile, 'w')

	xmo3=vmj0+dproJ
	xmo2=bmv0+xmo3
	xmo1=umb0+xmo2
	xmo4=xmo3-vmr0c
	xmo5=xmo3-vmr0j
	xmo6=xmo3-vmi0c
	xmo7=xmo3-vmi0j
	xmo8=dproJ
	xmo9=xmo3-vmh0
	xmo10=xmo3-vmk0
	xmo11=xmo3-vml0
	xmo12=xmo3-vmm0

	xmo=[xmo1,xmo2,xmo3,xmo4,xmo5,xmo6,xmo7,xmo8,xmo9,xmo10,xmo11,xmo12]
	xwef=[0.36,0.44,0.55,0.64,0.7,0.79,0.9,1.22,1.63,2.19,3.45,4.75]
	xzp=[1.81e-20,4.26e-20,3.64e-20,3.08e-20,3.01e-20,2.55e-20, \
		2.43e-20,1.57e-20,1.02e-20,6.36e-21,2.81e-21,1.54e-21]

	save_xfluxb=[]
	for i in range (0,12):
		xfnu=xzp[i]*10.**(-xmo[i]/2.5)
		xnu=c*1e4/xwef[i]
		xwangs=xwef[i]*1.e4
		xfl=xnu*xfnu/xwangs
		xfluxb=np.log10(xnu*xfnu)
		save_xfluxb.append(xfluxb)
	for i in range (0,nwl):
		if wl_phot[i] < xwef[11]:
			flux_phot=util.interp(wl_phot[i],xwef,save_xfluxb,12,i0)
			flux_phot=10**flux_phot
		else:
			flux_phot=(10**save_xfluxb[11])*((xwef[11]/wl_phot[i])**3)
		filephot.write('{}\t{}\n'.format(wl_phot[i],flux_phot[0]))
	filephot.close()

# calculate scaled template photosphere using PM13 colors
if calcphot=='yes' and table=='pm13':

	i0=0

	filephotwl=ascii.read(photfilewl,Reader=ascii.NoHeader,data_start=0,data_end=1)
	nwl=filephotwl['col1'][0]
	filephotwl=ascii.read(photfilewl,Reader=ascii.NoHeader,data_start=1)
	wl_phot=filephotwl['col1']

	filephot = open(photfile, 'w')

	xmo3=vmj0+dproJ
	xmo2=bmv0+xmo3
	xmo1=umb0+xmo2
	xmo4=xmo3-vmr0c
	xmo5=xmo3-vmi0c
	xmo6=dproJ
	xmo7=xmo3-vmh0
	xmo8=xmo3-vmk0
	xmo9=xmo8-kmw10
	xmo10=xmo8-kmw20
	xmo11=xmo8-kmw30
	xmo12=xmo8-kmw40
	
	xmo=[xmo1,xmo2,xmo3,xmo4,xmo5,xmo6,xmo7,xmo8,xmo9,xmo10,xmo11,xmo12]
	xwef=[0.36,0.44,0.55,0.64,0.79,1.22,1.63,2.19,3.35,4.60,11.56,22.09]
	# zero points - erg/cm2/s/Hz
	# from Bessell 1979,BB88 nir, Jarrett et al. 2011 WISE 
	# for Jarrett convert Fnu (Jy) using 1Jy=1e-23 erg/s/cm2/Hz
	xzp=[1.81e-20,4.26e-20,3.64e-20,3.08e-20,2.55e-20,1.57e-20,1.02e-20, \
	6.36e-21,3.10e-21,1.72e-21,3.17e-22,8.36e-23]

	save_xfluxb=[]
	for i in range (0,12):
		xfnu=xzp[i]*10.**(-xmo[i]/2.5)
		xnu=c*1e4/xwef[i]
		xwangs=xwef[i]*1.e4
		xfl=xnu*xfnu/xwangs
		xfluxb=np.log10(xnu*xfnu)
		save_xfluxb.append(xfluxb)
	for i in range (0,nwl):
		if wl_phot[i] < xwef[11]:
			flux_phot=util.interp(wl_phot[i],xwef,save_xfluxb,12,i0)
			flux_phot=10**flux_phot
		else:
			flux_phot=(10**save_xfluxb[11])*((xwef[11]/wl_phot[i])**3)
		filephot.write('{}\t{}\n'.format(wl_phot[i],flux_phot[0]))
	filephot.close()





