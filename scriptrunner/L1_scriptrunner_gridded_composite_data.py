#!/usr/bin/env python

import os


LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTL_TYPES = ["PICTL", "SSTPICTL"]

for LOC in LOCS:
    for PICTL_TYPE in PICTL_TYPES:
        args = " ".join([LOC, PICTL_TYPE])
        args_ = "_".join([LOC, PICTL_TYPE])
        job_file = "jobs/" + args_ + '.sbatch'
        open(job_file, 'a')

        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash -l\n")
            fh.writelines("#PBS -q casper\n")
            fh.writelines("#PBS -N gridded_composite\n")
            fh.writelines("#PBS -A P04010022\n")
            fh.writelines("#PBS -l select=1\n")
            fh.writelines("#PBS -l walltime=01:00:00\n")
            fh.writelines("#PBS -o out/gridded_composite.out\n")
            fh.writelines("#PBS -e out/gridded_composite.out\n")

            fh.writelines("module load ncarenv\n")
            fh.writelines("module load python\n")
            fh.writelines("ncar_pylib my_npl_clone_casper\n")
            fh.writelines("python ../py/L1/gridded_composite_data.py " + args + " > logs/gridded_composite_" + args_ + ".log\n")
            fh.writelines("deactivate\n")
            
        os.system("qsub %s" %job_file)


