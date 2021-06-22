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
            fh.writelines("#PBS -q casper\n")
            fh.writelines("#PBS -N JJA_mean\n")
            fh.writelines("#PBS -A P04010022\n")
            fh.writelines("#PBS -l select=1\n")
            fh.writelines("#PBS -l walltime=02:00:00\n")
            fh.writelines("#PBS -o out/JJA_mean.out\n")
            fh.writelines("#PBS -e out/JJA_mean_e.out\n")

            fh.writelines("module load ncarenv\n")
            fh.writelines("module load python\n")
            fh.writelines("ncar_pylib my_npl_clone_casper\n")
            fh.writelines("python ../py/L0/JJA_mean.py " + args + " > logs/JJA_mean_" + args_ + ".log\n")
            fh.writelines("deactivate\n")
            
        os.system("qsub %s" %job_file)


