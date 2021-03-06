#!/usr/bin/env python

import os


LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTLS = ["PICTL", "SSTPICTL"]
N_sets = [[150, 100, 20, 1798]]

for LOC in LOCS:
    for PICTL in PICTLS:
        for sets in N_sets:
            N = str(sets[0])
            N_S = str(sets[1])
            N_R = str(sets[2])
            N_Y = str(sets[3])
            args = " ".join([LOC, PICTL, N, N_S, N_R, N_Y])
            args_ = "_".join([LOC, PICTL, N, N_S, N_R, N_Y])
            job_file = "jobs/" + args_ + '.sbatch'
            open(job_file, 'a')

            with open(job_file, "w") as fh:
                fh.writelines("#!/bin/bash -l\n")
                fh.writelines("#PBS -q casper\n")
                fh.writelines("#PBS -N dyn_adj\n")
                fh.writelines("#PBS -A P04010022\n")
                fh.writelines("#PBS -l select=1:mem=100GB\n")
                fh.writelines("#PBS -l walltime=12:00:00\n")
                fh.writelines("#PBS -o out/dyn_adj.out\n")

                fh.writelines("module load ncarenv\n")
                fh.writelines("module load python\n")
                fh.writelines("ncar_pylib my_npl_clone_casper\n")
                fh.writelines("python ../py/L0/dyn_adj.py " + args + " > logs/dyn_adj_" + args_ + ".log\n")
                fh.writelines("deactivate\n")
                
            os.system("qsub %s" %job_file)


