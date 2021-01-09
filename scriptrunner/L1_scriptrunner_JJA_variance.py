#!/usr/bin/env python

import os

job_file = "jobs/JJA_variance.sbatch"
open(job_file, 'a')

with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#SBATCH --job-name=JJA_variance\n")
    fh.writelines("#SBATCH --account=P04010022\n")
    fh.writelines("#SBATCH --ntasks=1\n")
    fh.writelines("#SBATCH --time=1:00:00\n")
    fh.writelines("#SBATCH --partition=dav\n")
    fh.writelines("#SBATCH --output=out/JJA_variance.out.%j\n")
    fh.writelines("module load ncarenv\n")
    fh.writelines("module load python\n")
    fh.writelines("ncar_pylib my_npl_clone_casper\n")
    fh.writelines("python ../py/L1/JJA_variance.py "  + " > logs/JJA_variance"  + ".log\n")
    fh.writelines("deactivate\n")
    
os.system("sbatch %s" %job_file)


