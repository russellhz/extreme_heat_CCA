#!/usr/bin/env python

import os

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTLS = ['PICTL', 'SSTPICTL']
LAGS = [1,3]

for LOC in LOCS:
    for PICTL in PICTLS:
        for LAG in LAGS:
            args = " ".join([LOC,PICTL, str(LAG)])
            args_ = "_".join([LOC,PICTL, str(LAG)])

            job_file = "jobs/distance_matrix_" + args_ + '.sbatch'
            open(job_file, 'a')

            with open(job_file, "w") as fh:
                fh.writelines("#!/bin/bash -l\n")
                fh.writelines("#PBS -q casper\n")
                fh.writelines("#PBS -N distance_matrix_LAG\n")
                fh.writelines("#PBS -A P04010022\n")
                fh.writelines("#PBS -l select=1:mem=30GB\n")
                fh.writelines("#PBS -l walltime=03:00:00\n")
                fh.writelines("#PBS -o out/distance_matrix.out\n")
                
                fh.writelines("module load ncarenv\n")
                fh.writelines("module load python\n")
                fh.writelines("ncar_pylib my_npl_clone_casper\n")
                fh.writelines("python ../py/L0/distance_matrix_LAG.py "  + args + " > logs/distance_matrix_LAG" + args_ +".log\n")
                fh.writelines("deactivate\n")
                
            os.system("qsub %s" %job_file)


