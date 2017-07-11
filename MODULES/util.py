#!/usr/bin/env python

from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------
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

#-------------------------------------------------------------
def parameters_siess(tstar,lui,commonpath):
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
		mass=99
		age=99e6
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
def parameters_baraffe(tstar,lui,commonpath):
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
		mass=99
		age=99e6
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
def redd(av,wl,commonpath):
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
def reddhd(av,wl,commonpath):
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
def reddccm89(wl,r,commonpath):
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
def reddmcclureavgt8rv5p0(av,wl,commonpath):
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
def reddmcclureavgt8rv3p1(av,wl,commonpath):
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




