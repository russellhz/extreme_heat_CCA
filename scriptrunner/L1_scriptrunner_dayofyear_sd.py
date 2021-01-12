#!/usr/bin/env python

import os
PICTL_TYPES = ["PICTL", "SSTPICTL"]

for PICTL_TYPE in PICTL_TYPES:
    job_file = "jobs/dayofyear_sd_" + PICTL_TYPE + '.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH --job-name=dayofyear_sd\n")
        fh.writelines("#SBATCH --account=P04010022\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --time=2:00:00\n")
        fh.writelines("#SBATCH --partition=dav\n")
        fh.writelines("#SBATCH --output=out/dayofyear_sd.out.%j\n")
        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L1/dayofyear_sd.py "  + PICTL_TYPE + " > logs/dayofyear_sd_" + PICTL_TYPE  + ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("sbatch %s" %job_file)


