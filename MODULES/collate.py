import numpy as np
from astropy.io import fits
from astropy.io import ascii
from glob import glob
import pdb
import os
from glob import glob
from scipy import interpolate

#########################
## Read struc function ##
#########################

def readstruc(infile):
    '''
    Function that reads the temperature and density structure
    of the disk from the file fort14.*.irr.*.dat
    '''
    f = open(infile,'r')

    # We define the string to parse the file
    string = "irad,err,error="
    lines = f.readlines()
    indexes = [] # position of starting line for each radius
    for i,line in enumerate(lines):
        if string in line:
            indexes.append(i) # We save the line where the string was found
    rstar = float(lines[indexes[0]+1].split()[4])/1.5e13 #in au
    nrad = len(indexes)
    nz = indexes[1] - indexes[0] - 4 # there are four "dummy" lines for each radii
    table = []
    radii = []
    for ind in indexes:
        params = []
        #the radii
        radii.append(float(lines[ind+1].split()[0])*rstar) # in au
        #the values
        for j in range(nz):
            params.append(lines[ind+4+j].split())
        table.append(params) #size of table: [nrad][nz][number of parameters]
    table = np.array(table).astype(np.float)

    radii = np.array(radii).astype(np.float) # radii in au
    z = table[:,:,0]*rstar # heights (in au)
    p = table[:,:,1]       # pressure (in cgs)
    t = table[:,:,2]       # temperature (in K)
    rho = table[:,:,3]     # density (in g cm-3)
    epsbig = table[:,:,4]  # epsilon big
    eps    = table[:,:,5]  # epsilon

    return radii, z, p, t, rho, epsbig, eps

#########################
#### collate function ###
#########################

def collate(path, destination,jobnum=None, name=None, file_outputs=None, optthin=0, clob=0, fill=3, noextinct = 1, noangle = 0, nowall = 0, nophot = 0, noscatt = 1, notemp = 0, struc = 1, shock = 0):
    """
     collate.py

     PURPOSE:
            Organizes and stores flux and parameters from the D'Alessio
            disk/optically thin dust models and jobfiles in a fits
            file with a header.

     CALLING SEQUENCE:
            collate(path, destination, jobnum, name, file_names, [optthin=1], [clob=1], [high = 1], [noextinc = 1], [noangle = 1], [nowall = 1], [nophot = 1], [noscatt = 0])


     INPUTS:
            path: String of with path to location of jobfiles and model result
                  files. Both MUST be in the same location!

            Destination: String with where you want the fits file to be
                         sent after it's made

            jobnum: String or integer associated with a job number label end.

            name: String of the name of the object

            file_outputs: String of the name of the file with names of DIAD outputs.
            If this parameter is used, jobnum and name are not required.

     OPTIONAL KEYWORDS
            optthin: Set this value to 1 (or True) to run the optically thin dust
                     version of collate instead of the normal disk code. This will
                     also place a tag in the header.
            shock: Set this to 1 if the model that is being collated is a Calvet 1998 shock model.
            clob: Set this value to 1 (or True) to overwrite a currently existing
                  fits file from a previous run.
            fill: Set this value to the number of digits in your job file (default is 3).
                    NOTE: Not needed if run using file_outputs.
            nowall: Set this value to 1 (or True) if you do NOT want to include a wall file.
                    DEFAULT: 0
            noangle: Set this value to 1 (or True) if you do NOT want to to include a disk file
                     NOTE: You cannot perform the self extinction correction without the angle file. If this is set to 1, then
                     the noextin keyword will also be set to 1 automatically.
                     DEFAULT: 0
            nophot: Set this value to 1 (or True) if you do NOT want to include a photosphere file
                    DEFAULT: 0
            noextinct: Set this value to 1 (or True) if you do NOT want to apply extinction
                     to the inner wall and photosphere. Generally do not need to since observations are typically de-reddened.
            noscatt: !!!!! NOTE: THIS IS SET TO 1 BY DEFAULT !!!!!
                     Set this value to 1 (or True) if you do NOT want to include the scattered light file.
                     Set this value to 0 (or False) if you DO want to include the scattered light file
            notemp: Set this to 1 if you do NOT want to include the radial distribution of temperatures
            struc: Set this to 1 if you DO want to include the 2D structure of the disk.
     EXAMPLES:
            To collate a single model run for the object 'myobject' under the
            job number '001', use the following commands:

            from collate import collate
            path = 'Some/path/on/the/cluster/where/your/model/file/is/located/'
            name = 'myobject'
            dest = 'where/I/want/my/collated/file/to/go/'
            modelnum = 1
            collate(path, dest, jobnum = modelnum, name = name)

            Note that:
            modelnum = '001' will also work.

            collate.py cannot handle multiple models at once, and currently needs to be
            run in a loop. An example run with 100 optically thin dust models would
            look something like this:

            from collate import collate
            path = 'Some/path/on/the/cluster/where/your/model/files/are/located/'
            name = 'myobject'
            dest = 'where/I/want/my/collated/files/to/go/'
            for i in range(100):
                collate(path, dest, jobnum = i+1, name = name, optthin = 1)


     NOTES:

            For the most current version of collate and EDGE, please visit the github respository:
            https://github.com/danfeldman90/EDGE

            Collate corrects the flux from the star and the inner wall for extinction from
            the outer disk.

            Label ends for model results should of form objectname_001,

            For disk models, job file name convention is job001

            For optically thin dust, job file name convention is job_optthin001

            amax in the optthin model did not originally have an s after it. It is changed in
            the header file to have the s to be consistant with the disk models.

            The file_outputs file is created by the jobfile, and it contains the names
            of the DIAD output files. Its name convention is objectname_001_outputs.

     MODIFICATION HISTORY
     Dan Feldman, 19 Oct 2016, Removed numCheck for the zfill fix. Changed high input to fill.
     Connor Robinson, 12, Nov, 2015, Added parsing for MDOTSTAR in edge
     Connor Robinson, 6 Aug 2015, Added error handling, the FAILED key in the header, and the failCheck and head functions
     Connor Robinson, 30 July 2015, Added scattered light + ability to turn off components of the model
     Connor Robinson, 24 July 2015, Added extinction from the outer disk  + flag to turn it off
     Connor Robinson, 23 July 2015, Updated documentation and added usage examples
     Dan Feldman, 19 July 2015, added numCheck() and high kwarg to handle integer jobnums
     Dan Feldman, 25 June 2015, Improved readability.
     Connor Robinson, Dan Feldman, 24 June 2015, Finished all current functionality for use
     Connor Robinson 26 May 2015, Began work on optically thin disk code
     Connor Robinson, Dan Feldman, 22 May 2015, Wrote disk code in python
     Connor Robinson 3, Mar, 2015, Added the /nounderscore and /photnum flags
     Connor Robinson 6 Nov, 2014 First version uploaded to cluster

    """

    # Select way of reading names of DIAD outputs.
    if file_outputs:
        # If file_outputs is given, read list_files from it
        try:
            f = open(path+file_outputs, 'r')
        except IOError:
            print('COLLATE: MISSING file_outputs FILE FOR JOB NUMBER '+jobnum+', RETURNING...')
            return

        # Names of output files saved in list_files
        list_files = np.array(f.read().splitlines())
        f.close()

        # Read name and jobnum
        name = file_outputs.split('_')[0]
        jobnum = file_outputs.split('_')[1]

    else:
        # If file_names is not provided, then jobnum and name should be used
        if (jobnum is None) or (name is None):
            raise IOError('COLLATE: file_outputs WAS NOT USED BUT jobnum AND/OR name ARE EMPTY.')

        # Convert jobnum into a string:
        jobnum = str(jobnum).zfill(fill)

        # Use glob to get the list of output files
        list_files = np.array(glob(path+'*'+name+'_'+jobnum+'*'))

    # If working with optically thin models
    if optthin == True and shock == False:

        #Read in file
        job = 'job_optthin'+jobnum

        try:
            f = open(path+job, 'r')
        except IOError:
            print('COLLATE: MISSING OPTTHIN JOB NUMBER '+jobnum+', RETURNING...')
            return

        jobf  = f.read()
        f.close()

        #Define what variables to record
        sdparam = (['TSTAR', 'RSTAR', 'DISTANCIA', 'MUI', 'ROUT', 'RIN', 'TAUMIN', 'POWER',
                    'FUDGEORG', 'FUDGETROI', 'FRACSIL', 'FRACENT', 'FRACFORST', 'FRACAMC',
                    'AMAXS'])
        dparam = np.zeros(len(sdparam), dtype = float)

        #Read in the data associated with this model
        dataarr = np.array([])
        file = list_files[['fort16' in element for element in list_files]]
        failed = 0
        size = 0
        miss = 0
        try:
            size = os.path.getsize(file[0])
        except IndexError:
            print("COLLATE: WARNING IN JOB "+jobnum+": MISSING FORT16 FILE (OPTICALLY THIN DUST MODEL), ADDED 'FAILED' TAG TO HEADER")
            failed = True
            miss = 1
        if miss != 1 and size == 0:
            print("COLLATE: WARNING IN JOB "+jobnum+": EMPTY FORT16 FILE (OPTICALLY THIN DUST MODEL), ADDED FAILED TAG TO HEADER")
            failed = True

        if failed == False:
            data = ascii.read(file[0])
        #Combine data into a single array to be consistant with previous version of collate
            if size !=0:
                dataarr = np.concatenate((dataarr, data['col1']))
                dataarr = np.concatenate((dataarr, data['col3']))

        #If the file is missing/empty, add an empty array to collated file
        if failed != 0:
            dataarr = np.array([])

        #Convert anything that can't be read as a float into a nan
        tempdata = np.zeros(len(dataarr))
        floaterr = 0

        if failed == 0:
            for i, value in enumerate(dataarr):
                try:
                    tempdata[i] = float(dataarr[i]) #dataarr[i].astype(float)
                except ValueError:
                    floaterr = 1
                    tempdata[i] = float('nan')

            if floaterr == 1:
                print('COLLATE: WARNING IN JOB '+jobnum+': FILES CONTAIN FLOAT OVERFLOW/UNDERFLOW ERRORS, THESE VALUES HAVE BEEN SET TO NAN')

            axis_count = 2; #One axis for flux, one for wavelength
            dataarr = np.reshape(tempdata, (axis_count, int(len(tempdata)/axis_count)))

        #Make an HDU object to contain header/data
        hdu = fits.PrimaryHDU(dataarr)

        #Parse variables according to convention in job file
        for ind, param in enumerate(sdparam):

            #Handles the case of AMAXS which is formatted slightly differently
            if param  == 'AMAXS':
                for num in range(10):
                    if jobf.split("lamax='amax")[num].split("\n")[-1][0] == 's':
                        samax = jobf.split("lamax='amax")[num+1].split("'")[0]
                        if samax == '1mm':
                            hdu.header.set(param, 1000.)
                        else:
                            hdu.header.set(param, float(samax.replace('p', '.')))

            #Handle the rest of the variables
            else:
                paramold = param
                if param   == 'DISTANCIA':
                    param = 'DISTANCE' #Reduce the amount of Spanish here
                elif param == 'FUDGETROI':
                    param = 'FUDGETRO'
                elif param == 'FRACFORST':
                    param = 'FRACFORS'
                hdu.header.set(param, float(jobf.split("set "+paramold+"='")[1].split("'")[0]))

        hdu.header.set('OBJNAME', name)
        hdu.header.set('JOBNUM', jobnum)
        hdu.header.set('OPTTHIN', 1)
        hdu.header.set('WLAXIS', 0)
        hdu.header.set('LFLAXIS',1)

        if failed == 1:
            hdu.header.set('Failed', 1)

        hdu.writeto(destination+name+'_OTD_'+jobnum+'.fits', clobber = clob)

        if nowall == 1 or noangle == 1 or nophot == 1:
            print("COLLATE: WARNING IN JOB "+jobnum+": KEYWORDS THAT HAVE NO AFFECT ON OPTICALLY THIN DUST HAVE BEEN USED (NOPHOT, NOWALL, NOANGLE)")

    # If working with job models start here
    elif optthin == False and shock == False:

        #read in file
        job = 'job'+jobnum

        try:
            f = open(path+job, 'r')
        except IOError:
            print('COLLATE: MISSING JOB FILE '+jobnum+', RETURNING...')
            return

        jobf = f.read()
        f.close()

        #Check to see if the name + jobnum matches up with the labelend, if it doens't, return
        labelend = jobf.split("set labelend='")[1].split("'")[0]

        if labelend != name+'_'+jobnum:
            print('COLLATE: NAME IS NOT THE SAME AS THE NAME IN JOB '+jobnum+' LABELEND: '+labelend+', RETURNING...')
            return

        #Define what variables to record
        sparam = (['MSTAR', 'TSTAR', 'RSTAR', 'DISTANCIA','MDOT', 'MDOTSTAR','ALPHA', 'MUI', 'RDISK',
                   'AMAXS', 'EPS', 'ZTRAN', 'WLCUT_ANGLE', 'WLCUT_SCATT', 'NSILCOMPOUNDS', 'SILTOTABUN',
                   'AMORPFRAC_OLIVINE', 'AMORPFRAC_PYROXENE', 'FORSTERITE_FRAC', 'ENSTATITE_FRAC',
                   'TEMP', 'ALTINH', 'TSHOCK', 'AMAXW', 'AMAXB','D2G'])
        dparam = np.zeros(len(sparam), dtype = float)

        #Parse variables according to convention in the job file
        for ind, param in enumerate(sparam):
            if param == 'AMAXS':
                for split_amaxs in jobf.split("AMAXS='")[1:]:
                    if split_amaxs.split("\n")[1][0] == '#':
                        continue
                    elif split_amaxs.split("\n")[1][0] == 's':
                        dparam[ind] = float(split_amaxs.split("'")[0])
                    else:
                        raise IOError('COLLATE: FAILED ON AMAXS VALUE. FIX JOB FILE '+jobnum)

            elif param == 'AMAXW':
                for split_amaxw in jobf.split("AMAXW='")[1:]:
                    if split_amaxw.split("\n")[1][0] == '#':
                        continue
                    elif split_amaxw.split("\n")[1][0] == 's':
                        #Check if the wall has a different value than AMAXS. If not, assign it the value of AMAXS
                        if len(jobf.split("\nset AMAXW=$AMAXS")) > 1:
                            dparam[ind] = dparam[np.array(sparam) == 'AMAXS']
                        else:
                            dparam[ind] = float(split_amaxw.split("'")[0])
                    else:
                        raise IOError('COLLATE: FAILED ON AMAXW VALUE. FIX JOB FILE '+jobnum)

            elif param == 'TEMP' or param == 'TSHOCK':
                try:
                    dparam[ind] = float(jobf.split(param+"=")[1].split(".")[0])
                except ValueError:
                    raise ValueError('COLLATE: MISSING . AFTER '+param+' VALUE, GO FIX IN JOB FILE ' +jobnum)

            elif param == 'MDOTSTAR':
                #MDOTSTAR is set often set to $MDOT, but could also be set to a number
                #If it is the same as MDOT/not there, grab the value of MDOT
                try:
                    #Parse by " MDOTSTAR=' ", if it's a value will pick it out, if it's not there/$MDOT will throw value error.
                    dparam[ind] = float(jobf.split(param+"='")[1].split("'")[0])

                except IndexError:
                    dparam[ind] = dparam[sparam.index("MDOT")]
                    try:
                        nomdotstar = jobf.split(param+"=")[1]
                    except IndexError:
                        print('COLLATE: WARNING IN JOB '+jobnum+ ': NO VALUE FOR MDOTSTAR IN JOBFILE, ASSUMING MDOTSTAR = MDOT')

            else:
                dparam[ind] = float(jobf.split('set '+param+"='")[1].split("'")[0])

        #Rename header labels that are too long
        sparam[sparam.index('AMORPFRAC_OLIVINE')]  = 'AMORF_OL'
        sparam[sparam.index('AMORPFRAC_PYROXENE')] = 'AMORF_PY'
        sparam[sparam.index('WLCUT_ANGLE')] = 'WLCUT_AN'
        sparam[sparam.index('WLCUT_SCATT')] = 'WLCUT_SC'
        sparam[sparam.index('NSILCOMPOUNDS')] = 'NSILCOMP'
        sparam[sparam.index('SILTOTABUN')] = 'SILTOTAB'
        sparam[sparam.index('FORSTERITE_FRAC')] = 'FORSTERI'
        sparam[sparam.index('ENSTATITE_FRAC')] = 'ENSTATIT'

        #Reduce the amount of Spanish here
        sparam[sparam.index('DISTANCIA')] = 'DISTANCE'

        #Read in data from outputs (if the no____ flags are not set)

        #set up empty array to accept data, column names and axis number
        dataarr = np.array([])
        axis = {'WLAXIS':0}
        axis_count = 1 #Starts at 1, axis 0 reserved for wavelength information

        #Read in arrays and manage axis information
        #Also handles errors for missing/empty files

        failed = False;
        size = 0
        miss = 0

        if nophot == 0:
            photfile = list_files[['Phot' in element for element in list_files]][0]
            try:
                size = os.path.getsize(photfile)
            except IndexError:
                print("COLLATE: WARNING IN JOB "+jobnum+": MISSING PHOTOSPHERE FILE, ADDED 'FAILED' TAG TO HEADER. NOPHOT SET TO 1")
                nophot = 1
                failed = True
                miss = 1

            if miss != 1 and size != 0:
                phot  = ascii.read(photfile)
                axis['PHOTAXIS'] = axis_count
                dataarr = np.concatenate((dataarr, phot['col1']))
                dataarr = np.concatenate((dataarr, phot['col2']))
                axis_count += 1
            elif miss != 1 and size == 0:
                print("COLLATE: WARNING IN JOB "+jobnum+": PHOT FILE EMPTY, ADDED 'FAILED' TAG TO HEADER. NOPHOT SET TO 1")
                nophot = 1
                failed = True

        elif nophot != 1 and nophot != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOPHOT KEYWORD, SHOULD BE 1 OR 0')

        size = 0
        miss = 0

        if nowall == 0:
            wallfile = list_files[['fort17' in element for element in list_files]][0]
            try:
                size = os.path.getsize(wallfile)
            except IndexError:
                print("COLLATE: WARNING IN JOB "+jobnum+": MISSING FORT17 (WALL) FILE, ADDED 'FAILED' TAG TO HEADER. NOWALL SET TO 1")
                nowall = 1
                failed = True
                miss = 1

            if miss != 1 and size != 0:
                wall  =  ascii.read(wallfile, data_start = 9)
                axis['WALLAXIS'] = axis_count
                #If the photosphere was not run, then grab wavelength information from wall file
                if nophot != 0:
                    dataarr = np.concatenate((dataarr, wall['col1']))

                dataarr = np.concatenate((dataarr, wall['col2']))
                axis_count += 1

            elif miss != 1 and size == 0:
                print("COLLATE: WARNING IN JOB "+jobnum+": FORT17 (WALL) FILE EMPTY, ADDED 'FAILED' TAG TO HEADER. NOWALL SET TO 1")
                failed = True
                nowall = 1

        elif nowall != 1 and nowall != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOWALL KEYWORD, SHOULD BE 1 OR 0')

        miss = 0
        size = 0

        if noangle == 0:
            anglefile = list_files[['angle' in element for element in list_files]][0]
            try:
                size = os.path.getsize(anglefile)
            except IndexError:
                print("COLLATE: WARNING IN JOB "+jobnum+": MISSING ANGLE (DISK) FILE, ADDED 'FAILED' TAG TO HEADER. NOANGLE SET TO 1")
                noangle = 1
                failed = True
                miss = 1

            if miss != 1 and size != 0:
                try:
                    angle = ascii.read(anglefile, data_start = 1)
                    axis['ANGAXIS'] = axis_count

                    #If the photosphere was not run, and the wall was not run then grab wavelength information from angle file
                    if nophot != 0 and nowall != 0:
                        dataarr = np.concatenate((dataarr, angle['col1']))

                    dataarr = np.concatenate((dataarr, angle['col4']))
                    axis_count += 1

                except KeyError:
                    print("COLLATE: WARNING IN JOB "+jobnum+": ANGLE FILE VALUES EMPTY, ADDED 'FAILED' TAG TO HEADER. NOANGLE SET TO 1")
                    failed = True
                    noangle = 1

            elif miss != 1 and size == 0:
                print("COLLATE: WARNING IN JOB "+jobnum+": ANGLE (DISK) FILE EMPTY, ADDED 'FAILED' TAG TO HEADER. NOANGLE SET TO 1")
                failed = True
                noangle = 1

        elif noangle != 1 and noangle != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOANGLE KEYWORD, SHOULD BE 1 OR 0')

        miss = 0
        size = 0

        if noscatt == 0:
            scattfile = list_files[['scatt' in element for element in list_files]][0]
            try:
                size = os.path.getsize(scattfile)
            except IndexError:
                print("COLLATE: WARNING IN JOB "+jobnum+": MISSING SCATT FILE, ADDED 'FAILED' TAG TO HEADER. NOSCATT SET TO 1")
                noscatt = 1
                failed = True
                miss = 1

            if miss != 1 and size > 100:
                scatt = ascii.read(scattfile, data_start = 1)
                axis['SCATAXIS'] = axis_count
                    #If the photosphere, wall and disk were not run, then grab wavelength information from scatt file
                if nophot != 0 and nowall != 0 and noangle != 0:
                    dataarr = np.concatenate((dataarr, scatt['col1']))

                dataarr = np.concatenate((dataarr, scatt['col4']))
                axis_count += 1

            elif miss != 1 and size == 0 or miss != 1 and size < 100:
                print("COLLATE: WARNING IN JOB "+jobnum+": SCATT FILE EMPTY, ADDED 'FAILED' TAG TO HEADER. NOSCATT SET TO 1")
                failed = True
                noscatt = 1

        elif noscatt != 1 and noscatt != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOSCATT KEYWORD, SHOULD BE 1 OR 0')

        if noextinct == 0:
            if noangle != 0:
                print("COLLATE: WARNING IN JOB "+jobnum+": ANGLE (DISK) FILE "+jobnum+" REQUIRED FOR EXTINCTION FROM DISK. ADDED 'FAILED' TAG TO HEADER, NOEXTINCT SET TO 1")
                failed = 1
                noextinct = 1
            else:
                dataarr = np.concatenate((dataarr, angle['col6']))
                axis['EXTAXIS'] = axis_count
                axis_count += 1

        elif noextinct != 1 and noextinct != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOANGLE KEYWORD, SHOULD BE 1 OR 0')

        #if data has values that overflow/underflow float type, replace them with NaN dataarr = tempdata
        floaterr = 0
        tempdata = np.zeros(len(dataarr))

        for i, value in enumerate(dataarr):
            try:
                tempdata[i] = float(dataarr[i]) #dataarr[i].astype(float)
            except ValueError:
                floaterr = 1
                tempdata[i] = float('nan')

        if floaterr == 1:
            print('COLLATE: WARNING IN JOB '+jobnum+': FILES CONTAIN FLOAT OVERFLOW/UNDERFLOW ERRORS, THESE VALUES HAVE BEEN SET TO NAN')

        #Put data array into the standard form for EDGE
        dataarr = tempdata
        dataarr = np.reshape(dataarr, (axis_count, int(len(dataarr)/axis_count)))

        #Self extinct the photosphere and wall
        if noextinct == 0:
            if nophot == 0:
                dataarr[axis['PHOTAXIS'],:] *=np.exp((-1)*dataarr[axis['EXTAXIS'],:])
            if nowall == 0:
                dataarr[axis['WALLAXIS'],:] *=np.exp((-1)*dataarr[axis['EXTAXIS'],:])


        #Create the header and add parameters
        hdu = fits.PrimaryHDU(dataarr)

        #Add a few misc tags to the header
        hdu.header.set('OBJNAME', name)
        hdu.header.set('JOBNUM', jobnum)

        for i, param in enumerate(sparam):
            hdu.header.set(param, dparam[i])


        if nowall != 1:
            hdu.header.set('RIN', float(np.loadtxt(list_files[['rin.t' in element for element in list_files]][0])))

        #Get the disk mass
        if noangle != 1:
            try:
                massfile = np.genfromtxt(list_files[['fort15' in element for element in list_files]][0], skip_header = 3,skip_footer = 1)
                diskmassrad = massfile[:,0]
                diskmassvals = massfile[:,10]
                massfit = interpolate.interp1d(diskmassrad, diskmassvals)
                if nowall != 1:
                    hdu.header.set('DISKMASS', float(massfit(hdu.header['RDISK']))-float(massfit(hdu.header['RIN'])))
                else:
                    print("COLLATE:WARNING IN JOB "+jobnum+": nowall = 1, SO THE DISK MASS WILL ALSO HAVE THE MASS INSIDE THE CAVITY.")
                    hdu.header.set('DISKMASS', float(massfit(hdu.header['RDISK'])))

            except:
                print("COLLATE:WARNING IN JOB "+jobnum+": DISK MASS CALCULATION FAILED.")
                hdu.header.set('DISKMASS','FAILED')
        else:
            hdu.header.set('DISKMASS','FAILED')

        #Create tags in the header that match up each column to the data enclosed]
        for naxis in axis:
            hdu.header.set(naxis, axis[naxis])

        #Add a tag to the header if the noextinct flag is on
        if noextinct == 1:
            hdu.header.set('NOEXT', 1)

        #Get the Temperature structure data from the prop file
        if notemp == 0:
            propfile = list_files[['prop' in element for element in list_files]][0]
            try:
                size = os.path.getsize(propfile)
                miss = 0
            except IndexError:
                print("COLLATE: WARNING IN JOB "+jobnum+": MISSING PROP (PROPERTIES) FILE, ADDED 'FAILED' TAG TO HEADER. NOTEMP SET TO 1")
                failed = True
                miss = 1
                hdu.header.set('NOTEMP', 1)

            if miss != 1 and size != 0:
                try:
                    propdatatable = ascii.read(propfile, data_start = 1)
                    propdata = np.zeros([len(propdatatable[0]),len(propdatatable)])
                    #Replace 'D' in the table with 'e' and convert into a numpy array the terrible brute force way
                    for i, column in enumerate(propdatatable):
                        for j, value in enumerate(column):
                            propdata[j,i] = np.float(str.replace(value, 'D', 'e'))

                    #Start making a new array that contains structural data
                    sdata = 10**(np.vstack([propdata[0,:], propdata[1,:], propdata[2,:], propdata[4,:], propdata[5,:], propdata[11,:], propdata[12,:]]))
                    saxisnames = ['RADIUS', 'TEFF', 'TIRR', 'TZEQ0', 'TZEQZMAX', 'TZEQZS', 'TZTAUS']

                    #Add a separate fits extension to contain structural data
                    saxiscounter = 0
                    shdu = fits.ImageHDU(sdata)

                    for saxis in saxisnames:
                        shdu.header.set(saxis, saxiscounter)
                        saxiscounter +=1

                except IndexError:
                    print("COLLATE: WARNING IN JOB "+jobnum+": PROP FILE FOUND, BUT APPEARS TO HAVE FAILED. ADDED 'FAILED' TAG TO HEADER. NOTEMP SET TO 1")
                    failed = True
                    hdu.header.set('NOTEMP', 1)
            elif miss !=1 and size ==0:
                print("WARNING IN JOB "+jobnum+": PROP (PROPERTIES) FILE EMPTY, ADDED 'FAILED' TAG TO HEADER. NOTEMP SET TO 1")
                failed = True
                nowall = 1
                hdu.header.set('NOTEMP', 1)
        elif notemp == 1:
            hdu.header.set('NOTEMP', 1)
        elif notemp != 1 and notemp != 0:
            raise IOError('COLLATE: INVALID INPUT FOR NOTEMP KEYWORD, SHOULD BE 1 OR 0')

        #Add FAILED tag to header if any of the model elements were not found
        if failed == 1:
            hdu.header.set('FAILED', 1)

        ##### Structure of disk #####
        if struc == 1:
            # In order to avoid opening the fort14.vis..., we do this in two steps
            strucfiles = list_files[['fort14' in element for element in list_files]]
            strucfile = strucfiles[['irr' in element for element in strucfiles]][0]
            radii, z, p, t, rho, epsbig, eps = readstruc(strucfile) # arrays as [nrad,nz]

            # We save them in different fits extensions as [nz,nrad]
            radii_hdu = fits.ImageHDU(np.meshgrid(radii,z[0,:])[0])
            radii_hdu.header.set('COMMENT','Radii (au)')
            z_hdu = fits.ImageHDU(z.T)
            z_hdu.header.set('COMMENT','Z (au)')
            p_hdu = fits.ImageHDU(p.T)
            p_hdu.header.set('COMMENT','Pressure (cgs)')
            t_hdu = fits.ImageHDU(t.T)
            t_hdu.header.set('COMMENT','Temperature (K)')
            rho_hdu = fits.ImageHDU(rho.T)
            rho_hdu.header.set('COMMENT','Density (g cm-2)')
            epsbig_hdu = fits.ImageHDU(epsbig.T)
            epsbig_hdu.header.set('COMMENT','Epsbig')
            eps_hdu = fits.ImageHDU(eps.T)
            eps_hdu.header.set('COMMENT','Eps')

        #We add the different fits extensions
        if struc == 1:
            if notemp == 0:
                hdu.header.set('EXTS',9)
                hdu.header.set('SED',1)
                hdu.header.set('RADSTRUC',2)
                hdu.header.set('RAD_EXT',3)
                hdu.header.set('Z_EXT',4)
                hdu.header.set('P_EXT',5)
                hdu.header.set('T_EXT',6)
                hdu.header.set('RHO_EXT',7)
                hdu.header.set('EPSB_EXT',8)
                hdu.header.set('EPS_EXT',9)
                hdu_list = fits.HDUList([hdu,shdu,radii_hdu,z_hdu,p_hdu,t_hdu,rho_hdu,epsbig_hdu,eps_hdu])
            else:
                hdu.header.set('EXTS',8)
                hdu.header.set('SED',1)
                hdu.header.set('RAD_EXT',2)
                hdu.header.set('Z_EXT',3)
                hdu.header.set('P_EXT',4)
                hdu.header.set('T_EXT',5)
                hdu.header.set('RHO_EXT',6)
                hdu.header.set('EPSB_EXT',7)
                hdu.header.set('EPS_EXT',8)
                hdu_list = fits.HDUList([hdu,radii_hdu,z_hdu,p_hdu,t_hdu,rho_hdu,epsbig_hdu,eps_hdu])
        else:
            if notemp == 0:
                hdu.header.set('EXTS',2)
                hdu.header.set('SED',1)
                hdu.header.set('RADSTRUC',2)
                hdu_list = fits.HDUList([hdu,shdu])
            else:
                hdu.header.set('EXTS',1)
                hdu.header.set('SED',1)
                hdu_list = hdu

        #Write fits file
        hdu_list.writeto(destination+name+'_'+jobnum+'.fits', clobber = clob)

    #Now handle the shock file case
    elif shock == True and optthin == False:
        #Read in file
        job = 'job'+jobnum
        try:
            f = open(path+job, 'r')
        except IOError:
            print('COLLATE: MISSING SHOCK JOB NUMBER '+jobnum+', RETURNING...')
            return

        jobf = f.read()
        f.close()

        #Choose the variables you want to extract
        sparam = np.array(['DISTANCE', 'MASS', 'RADIO', 'TSTAR', 'BIGF', 'FILLING', 'WTTS','VEILING'])
        dparam = []

        #Begin extracting parameters
        for i, param in enumerate(sparam):

            #Handle the special cases of WTTS
            if param == 'WTTS':
                wtts = jobf.split("filewtts="+name+'_')[1].split("_")[0]
                dparam.append(wtts)

            #Handle the special cases of BIGF
            elif param == 'BIGF':
                dparam.append(jobf.split(param+"='")[1].split("'")[0])

            #Handle everything else
            else:
                dparam.append(float(jobf.split(param+"='")[1].split("'")[0]))


        #Add in the data for each column
        #set up empty array to accept data, column names and axis number
        dataarr = np.array([])
        axis = {'WLAXIS':0}
        axis_count = 1 #Starts at 1, axis 0 reserved for wavelength information

        #Load in the model
        datastart = 119
        footer = 8

        #Load in the data
        try:
            data = np.genfromtxt(path+'fort30.'+name+jobnum, skip_header = datastart, usecols = [1,2,3,4], skip_footer = footer)
        except StopIteration:
            print('COLLTE: MODEL '+jobnum+' FAILED, RETURNING...')
            return

        #Convert data into erg cm^-1 s^-1
        wl = data[:,0]
        Fhp = data[:,1]*data[:,0]
        Fpre = data[:,2]*data[:,0]
        Fphot = data[:,3]*data[:,0]

        #Add in axis labels
        axis['HEATAXIS'] = 1
        axis['PREAXIS'] = 2
        axis['PHOTAXIS'] = 3

        #Get the data in the right shape
        dataarr = np.vstack([wl, Fhp, Fpre, Fphot])

        #Create the header and add parameters
        hdu = fits.PrimaryHDU(dataarr)

        #Add a few misc tags to the header
        hdu.header.set('OBJNAME', name)
        hdu.header.set('JOBNUM', jobnum)

        #Add the rest of the parameters to the header
        for i, param in enumerate(sparam):
            hdu.header.set(param, dparam[i])

        #Create tags in the header that match up each column to the data enclosed]
        for naxis in axis:
            hdu.header.set(naxis, axis[naxis])

        #Write out the file
        hdu.writeto(destination+name+'_'+jobnum+'.fits', clobber = clob)

    # If you don't give a valid input for the optthin keyword, raise an error
    else:
        raise IOError('COLLATE: INVALID INPUT FOR OPTTHIN KEYWORD, SHOULD BE 1 OR 0')

    return

def failCheck(name, path = '', jobnum = 'all', fill = 3, optthin = 0):
    """

    PURPOSE:
        Opens up each header, checks if 'FAILED' tag = 1 and records the job number in a list if it is

    INPUTS:
        name: String of the name of object

    OPTIONAL INPUTS:
        path: Path to the collated file. Default is the current directory

        jobnum: Job number of object. Can be either a string or an int. If it's not set, failCheck
                will return ALL collated jobs that failed in the path directory

    KEYWORDS:
        optthin: Set this to 1 if the collated file is an optically thin dust file
        fill: Set this value to the number of digits in your job file (default is 3).

    OUTPUT
        Returns a list of failed jobs. If none are found, array will be empty.
    """

    opt = ''
    if optthin == 1:
        opt = 'OTD_'

    #Set up wildcards depending on number formating
    wildhigh = '?'*fill

    if jobnum == 'all':
        if optthin == 1:
            files = glob(path+name+'_'+opt+'*.fits')
        if optthin == 0:
            files = glob(path+name+'_'+wildhigh+'.fits')

        failed = []

        for file in files:
            HDU = fits.open(file)
            nofail = 0
            try:
                HDU[0].header['Failed'] == 1
            except KeyError:
                nofail = 1
            if nofail != 1:
                failed.append(file)

    if jobnum != 'all':

        if type(jobnum) == int:
            jobnum = str(jobnum).zfill(fill)

        failed = []
        nofail = 0

        file = glob(path+name+'_'+opt+jobnum+'.fits')

        try:
            HDU = fits.open(file[0])
        except IndexError:
            print('NO FILE MATCHING THOSE CRITERIA COULD BE FOUND, RETURNING...')
            return
        try:
            HDU[0].header['Failed'] == 1
        except KeyError:
            nofail = 1
        if nofail != 1:
            failed = [file[0]]

    return failed


def head(name, jobnum, path='', optthin = 0, fill = 3):
    """

    collate.head

    prints out the contents of the header of a collated file

    INPUTS:
        name: String of the name of object
        jobnum: Job number of object. Can be either a string or an int

    OPTIONAL INPUTS:
        path: Path to the collated file. Default is the current directory

    KEYWORDS:
        optthin: Set this to 1 If the collated file is an optically thin dust file
        fill: Set this value to the number of digits in your job file (default is 3).

    OUTPUTS:
        Prints the contents of the header to the terminal. Returns nothing else.

    """
    if type(jobnum) == int:
        jobnum = str(jobnum).zfill(fill)

    if optthin == 1:
        otd = 'OTD_'
    else:
        otd = ''

    file = path+name+'_'+otd+jobnum+'.fits'

    HDU = fits.open(file)

    print(repr(HDU[0].header))

def masscollate(name, destination = '',path = '', jobnum = None, optthin=0, clob=0, fill=3, noextinct = 1, noangle = 0, nowall = 0, nophot = 0, noscatt = 1, notemp = 0, shock= 0):
    '''
    collate.masscollate

    PURPOSE:
        Collates lots of files in a given directory

    INPUTS:
        name: Name of the object

    OPTIONAL INPUTS:
        jobnum: List/numpy array of the specfic job numbers that you want to collate. Default is 'None', which will collate all
                the jobs in the folder specified by path.
        path: Path to the all of the output from diad. Default is the current directory.
        destination: Path to where you want the collated files to go. Default is the current directory.

    KEYWORDS:
        Rest of the optional keywords are the same as collate

    '''

    #Get the list of job numbers if there are no specific models
    if np.all(jobnum) == None:
        if optthin == 1:
            files = glob(path+'job_optthin'+'?'*fill)
        else:
            files = glob(path+'job'+'?'*fill)

        #If there are no files found, report it and return
        if len(files) == 0:
            if path == '':
                print('NO JOB FILES FOUND IN THE CURRENT WORKING DIRECTORY. RETURNING...')
                return
            else:
                print('NO JOB FILES FOUND IN '+path+' RETURNING...')
                return

        jobnum = [x[-fill:] for x in files]

    #Collate the files
    for job in jobnum:
        collate(path, destination, jobnum=job, name=name, optthin=optthin, clob=clob, fill=fill, noextinct = noextinct, noangle = noangle, nowall = nowall, nophot = nophot, noscatt = noscatt, notemp = notemp, shock = shock)
