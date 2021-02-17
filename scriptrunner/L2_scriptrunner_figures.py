#!/usr/bin/env python

import os

job_file = "jobs/figures.sbatch"
open(job_file, 'a')

with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#SBATCH --job-name=figures\n")
    fh.writelines("#SBATCH --account=P04010022\n")
    fh.writelines("#SBATCH --ntasks=1\n")
    fh.writelines("#SBATCH --time=1:00:00\n")
    fh.writelines("#SBATCH --partition=dav\n")
    fh.writelines("#SBATCH --output=out/figures.out.%j\n")
    fh.writelines("module load ncarenv\n")
    fh.writelines("module load python\n")
    fh.writelines("ncar_pylib my_npl_clone_casper\n")
    fh.writelines("python ../py/L2/figures.py "  + " > logs/figures"  + ".log\n")
    fh.writelines("deactivate\n")
    
os.system("sbatch %s" %job_file)


