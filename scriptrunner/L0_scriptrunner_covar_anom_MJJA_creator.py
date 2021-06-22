#!/usr/bin/env python

import os

VARS = ["lhflx", "shflx", "soilwater_10cm", "ts"]  

for VAR in VARS:
    job_file = "jobs/" + 'covar_anom_MJJA_creator_'+ VAR +'.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#PBS -q casper\n")
        fh.writelines("#PBS -N covar_anom\n")
        fh.writelines("#PBS -A P04010022\n")
        fh.writelines("#PBS -l select=1:mem=100GB\n")
        fh.writelines("#PBS -l walltime=04:00:00\n")
        fh.writelines("#PBS -o out/covar_anom.out\n")
        fh.writelines("#PBS -e out/covar_anom_e.out\n")

        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L0/covar_anom_MJJA_creator.py " + VAR + " > logs/covar_anom_MJJA_creator_" + VAR + ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("qsub %s" %job_file)


