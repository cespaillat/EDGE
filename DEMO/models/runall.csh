#!/bin/csh

##next line: job name 
#$ -N joball
##next line: carries over environment variables
#$ -V

##send email to; for abort, begin, end; merge output and error
#$ -j y

##change to run directory
cd /projectnb/bu-disks/connorr/DEMO/

##and run job.
set my_job = `printf "job%03d" $SGE_TASK_ID`
csh $my_job

#run with
#qsub -t 1-3 runall.csh

wait

