#!/usr/bin/env python

import os


PICTL_TYPES = ["PICTL", "SSTPICTL"]
VARS = ['soilwater_10cm']

for VAR in VARS:
    for PICTL_TYPE in PICTL_TYPES:
        

        args = " ".join([VAR, PICTL_TYPE])
        args_ = "_".join([VAR, PICTL_TYPE])
        job_file = "jobs/" + 'JJA_mean' + '.sbatch'
        open(job_file, 'a')

        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash -l\n")
            fh.writelines("#SBATCH --job-name=JJA_mean\n")
            fh.writelines("#SBATCH --account=P04010022\n")
            fh.writelines("#SBATCH --ntasks=1\n")
            fh.writelines("#SBATCH --time=2:00:00\n")
            fh.writelines("#SBATCH --partition=dav\n")
            fh.writelines("#SBATCH --output=out/JJA_mean.out.%j\n")
            fh.writelines("module load ncarenv\n")
            fh.writelines("module load python\n")
            fh.writelines("ncar_pylib my_npl_clone_casper\n")
            fh.writelines("python ../py/L0/JJA_mean.py " + args + " > logs/JJA_mean_" + args_ + ".log\n")
            fh.writelines("deactivate\n")
            
        os.system("sbatch %s" %job_file)


