import numpy as np
import matplotlib.pyplot as plt
import EDGE as edge
import collate as c
import pdb
from astropy.io import fits

'''
DEMO_analysis_imlup.py

PURPOSE:
    Script that loads in data and models for IM Lup, and then finds the model/wall combination with the lowest chi^2
    
INPUTS:
    In order for this script to run properly, you will need to change the paths to the correct directory.
    If using this file as a template for other objects, it is likely that you will need to change it according to your needs.
    
    If you want to save the plot that this script creates, set 'save' to True.
    
OUTPUTS:
    Produces a plot with the model/wall with the lowest chi^2, along with a list of all the chi^2 + model numbers and best fitting wall heights.
    
NOTES:
    This script is supposed to act as a simple example of how to use EDGE for analysis, but is definitely not the rule. Significant changes will
    likely need to be made in order to analyze your objects.

AUTHOR:
    Connor Robinson, June 19th, 2017
    
'''

#Define the object name
obj = 'imlup'

#Set up paths. YOU WILL NEED TO CHANGE THIS!!!
picklepath = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/data/'
modelpath  = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/models/'
figpath    = '/Users/Connor/Desktop/Research/diad/EDGE/DEMO/'

#Set to True if you want to save the plot as a pdf
save = False

#-------------------------------------------------
#For the purposes of this example, you are not required to change anything below this line
#However, you should be able to understand what the code is doing before doing your own analysis
#-------------------------------------------------

#Define the jobs
jobs = np.arange(19)+1

#Define list of wall heights to try
altinh = [1,2,3,4,5]

#Load in the data pickle
targ = edge.loadPickle(obj, picklepath = picklepath)

#Create a blank list to append onto later
chi2 = []

#Begin looping over each job
for job in jobs:
    #Convert the job number into the right format. In this case, using a fill of 3
    job = str(job).zfill(3)
    
    #Load in the header. It will be used to check if jobs have failed.
    hdu = fits.open(modelpath+obj+'_'+job+'.fits')
    
    #Load innthe model
    model = edge.TTS_Model(obj, job, dpath = modelpath)
    
    #Check to see if the model failed and if it did, move onto the next model.
    try:
        failed = hdu[0].header['FAILED']
        pass
    except KeyError:
        
        #Create a black array to append onto later for fitting the best wall height
        chiwall = []
        
        #Initialize the model. For a pre-transitional disk, this command would be more complicated
        model.dataInit()
        
        #Loop over each wall height to find the best fitting wall
        for alt in altinh:
            #Calculate the total emission from all the components of the disk + star
            model.calc_total(altinh = alt, verbose = 0)
            
            #If you are running your code with the filter deconvolution, uncomment this
            #model.calc_filters(obj = targ)
            
            #Append the chi2 vlaue and the height of the wall
            chiwall.append([alt, edge.model_rchi2(targ, model)])
        
        #Convert the list into an array
        chiwall = np.array(chiwall)
        #Find the best fitting wall based on its chi^2
        bestwall = chiwall[np.argmin(chiwall[:,1])]
        
        #Now that the best wall has been found, use 
        model.dataInit()
        model.calc_total(altinh = bestwall[0],verbose = 0)
        
        chi2.append([float(job), edge.model_rchi2(targ, model), bestwall[0]])
        print(job)

chi2 = np.array(chi2)
order = np.argsort(chi2[:,1])

model = edge.TTS_Model(obj, int(chi2[order[0]][0]), dpath = modelpath)
model.dataInit()
model.calc_total(altinh = chi2[order[0]][2], verbose = 0)

edge.look(targ, model, jobn = int(chi2[order[0]][0]), ylim = [4e-13, 7e-9], save = save)






