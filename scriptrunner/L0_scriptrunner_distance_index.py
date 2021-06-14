#!/usr/bin/env python

import os

job_file = "jobs/distance_index.sbatch"
open(job_file, 'a')

with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash -l\n")
    fh.writelines("#PBS -q casper\n")
    fh.writelines("#PBS -N distance_index\n")
    fh.writelines("#PBS -A P04010022\n")
    fh.writelines("#PBS -l select=1\n")
    fh.writelines("#PBS -l walltime=01:00:00\n")
    fh.writelines("#PBS -o out/distance_index.out\n")
    fh.writelines("module load ncarenv\n")
    fh.writelines("module load python\n")
    fh.writelines("ncar_pylib my_npl_clone_casper\n")
    fh.writelines("python ../py/L0/distance_index.py "  + " > logs/distance_index.log\n")
    fh.writelines("deactivate\n")
    
os.system("qsub %s" %job_file)


