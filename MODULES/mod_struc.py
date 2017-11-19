#!/usr/bin/env python

import numpy as np
import os
import math
from collate import readstruc
from glob import glob

def gap_creator(labelend, inter_r = [], rho_deltas = [], temp_deltas = [],
    epsbig_deltas = [], eps_deltas = []):
    '''
    Function that modifies the structure of a disk model saved in
    the fort14.irr file. The modified fort14 is saved into a fort14.mod file.

    INPUTS:
    - labelend: labelend of the model, used to find the correct fort14.irr
    - inter_r: list of radii corresponding to the outer edges of the intervals in
    radius that want to be modified.
    - rho_deltas: list of deltas by which the density will be multiplied in the
    radius intervals provided.
    - temp_deltas: list of deltas for temperature.
    - epsbig_deltas: list of deltas for epsilon big.
    - eps_deltas: list of deltas for epsilon.

    OUTPUTS:
    A modified fort14.mod file with the new structure of the disk.

    EXAMPLE:
    - inter_r = [10.]; rho_deltas = [0.1]:
    Between 0 and 10 au the density will be multiplied by 0.1.
    - inter_r = [5.,20.]; rho_deltas =  [1.0,0.1]:
    Between 0 and 5 au the density will not be modified (multiplied by 1).
    Between 5 and 20 au the density will be multiplied by 0.1.
    '''
    if len(inter_r) == 0:
        print('GAP_CREATOR: No modification provided.')
        return
    if rho_deltas:
        if len(inter_r) != len(rho_deltas):
            raise IOError('GAP_CREATOR: Length of inter_r should be the same as rho_deltas')
    if temp_deltas:
        if len(inter_r) != len(temp_deltas):
            raise IOError('GAP_CREATOR: Length of inter_r should be the same as temp_deltas')
    if epsbig_deltas:
        if len(inter_r) != len(epsbig_deltas):
            raise IOError('GAP_CREATOR: Length of inter_r should be the same as epsbig_deltas')
    if eps_deltas:
        if len(inter_r) != len(eps_deltas):
            raise IOError('GAP_CREATOR: Length of inter_r should be the same as eps_deltas')

    files = glob('fort14*irr.'+labelend+'*')
    if len(files) == 0:
        raise IOError('GAP_CREATOR: fort14.irr file not found for '+labelend)
    elif len(files) > 1:
        raise IOError('GAP_CREATOR: More than one fort14.irr file found for '+labelend)

    fort14 = files[0]
    radii, z, p, t, rho, epsbig, eps = readstruc(fort14) # arrays as [nrad,nz]
    radii = np.meshgrid(radii,z[0,:])[0].T

    # We modify the density according to the given intervals and deltas
    r_0 = 0.0
    for i,r in enumerate(inter_r):
        if rho_deltas:
            rho[(r_0 < radii) & (radii <= r)] *= rho_deltas[i]
        if temp_deltas:
            t[(r_0 < radii) & (radii <= r)] *= temp_deltas[i]
        if epsbig_deltas:
            epsbig[(r_0 < radii) & (radii <= r)] *= epsbig_deltas[i]
        if eps_deltas:
            eps[(r_0 < radii) & (radii <= r)] *= eps_deltas[i]
        r_0 = r

    # We read the file to find where each radius begins
    f = open(fort14,'r')
    # We define the string to parse the file
    string = "irad,err,error="
    lines = f.readlines()
    indexes = [] # position of starting line for each radius
    for i,line in enumerate(lines):
        if string in line:
            indexes.append(i) # We save the line where the string was found
    f.close()
    rstar = float(lines[indexes[0]+1].split()[4])/1.5e13 #in au
    nrad = len(indexes)
    nz = indexes[1] - indexes[0] - 4 # there are four "dummy" lines for each radii

    # We save the new structure
    f = open(fort14.replace('irr','mod'),'w')
    # First dummy line
    f.write(lines[0])
    for i,ind in enumerate(indexes):
        # Each radius starts with four lines
        f.write(lines[ind])
        f.write(lines[ind+1])
        f.write(lines[ind+2])
        f.write(lines[ind+3])
        for j in range(nz):
            f.write("%f %g %g %g %g %f \n" %(z[i,j]/rstar,p[i,j],t[i,j],rho[i,j],epsbig[i,j],eps[i,j]))
    f.close()

    return
