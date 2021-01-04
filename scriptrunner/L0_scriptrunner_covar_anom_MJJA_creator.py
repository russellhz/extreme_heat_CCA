#!/usr/bin/env python

import os

VARS = ["lhflx", "shflx", "soilwater_10cm"]  

for VAR in VARS:
    job_file = "jobs/" + 'covar_anom_MJJA_creator_'+ VAR +'.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH --job-name=covar_anom\n")
        fh.writelines("#SBATCH --account=P04010022\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --time=04:00:00\n")
        fh.writelines("#SBATCH --partition=dav\n")
        fh.writelines("#SBATCH --mem=100G\n")
        fh.writelines("#SBATCH --output=out/covar_anom.out.%j\n")
        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L0/covar_anom_MJJA_creator.py " + VAR + " > logs/covar_anom_MJJA_creator_" + VAR + ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("sbatch %s" %job_file)


