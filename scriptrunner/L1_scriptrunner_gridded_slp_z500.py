#!/usr/bin/env python

import os

job_file = "jobs/gridded_slp_z500.sbatch"
open(job_file, 'a')

with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#SBATCH --job-name=gridded_slp_z500\n")
    fh.writelines("#SBATCH --account=P04010022\n")
    fh.writelines("#SBATCH --ntasks=1\n")
    fh.writelines("#SBATCH --time=1:00:00\n")
    fh.writelines("#SBATCH --partition=dav\n")
    fh.writelines("#SBATCH --output=out/gridded_slp_z500.out.%j\n")
    fh.writelines("module load ncarenv\n")
    fh.writelines("module load python\n")
    fh.writelines("ncar_pylib my_npl_clone_casper\n")
    fh.writelines("python ../py/L1/gridded_slp_z500.py "  + " > logs/gridded_slp_z500"  + ".log\n")
    fh.writelines("deactivate\n")
    
os.system("sbatch %s" %job_file)


