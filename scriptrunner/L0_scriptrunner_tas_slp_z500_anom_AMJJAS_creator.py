#!/usr/bin/env python

import os

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  

for LOC in LOCS:
    job_file = "jobs/" + 'anom_AMJJAS_creator_'+ LOC +'.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#PBS -q casper\n")
        fh.writelines("#PBS -N anom_AMJJAS\n")
        fh.writelines("#PBS -A P04010022\n")
        fh.writelines("#PBS -l select=1:mem=100GB\n")
        fh.writelines("#PBS -l walltime=04:00:00\n")
        fh.writelines("#PBS -o out/anom_AMJJAS.out\n")
        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L0/tas_slp_z500_anom_AMJJAS_creator.py " + LOC + " > logs/anom_AMJJAS_creator_" + LOC+ ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("sbatch %s" %job_file)


