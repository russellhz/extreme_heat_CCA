#!/usr/bin/env python

import os
#PICTL_TYPES = ["PICTL", "SSTPICTL"]

#for PICTL_TYPE in PICTL_TYPES:
job_file = "jobs/cesmle_slp_correlation"  + '.sbatch'
open(job_file, 'a')

with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#PBS -q casper\n")
    fh.writelines("#PBS -N cesmle_slp_correlation\n")
    fh.writelines("#PBS -A P04010022\n")
    fh.writelines("#PBS -l select=1\n")
    fh.writelines("#PBS -l walltime=01:00:00\n")
    fh.writelines("#PBS -o out/cesmle_slp_correlation.out\n")
    fh.writelines("#PBS -e out/cesmle_slp_correlation_e.out\n")

    fh.writelines("module load ncarenv\n")
    fh.writelines("module load python\n")
    fh.writelines("ncar_pylib my_npl_clone_casper\n")
    fh.writelines("python ../py/L1/cesmle_slp_correlation_LAG.py "  + " > logs/cesmle_slp_correlation"  + ".log\n")
    fh.writelines("deactivate\n")
    
os.system("qsub %s" %job_file)


