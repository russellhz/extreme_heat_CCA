#!/usr/bin/env python

import os

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  

for LOC in LOCS:
    job_file = "jobs/" + 'tas_slp_anom_AMJJAS_creator_'+ LOC +'.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH --job-name=tas_anom_AMJJAS\n")
        fh.writelines("#SBATCH --account=P04010022\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --time=03:00:00\n")
        fh.writelines("#SBATCH --partition=dav\n")
        fh.writelines("#SBATCH --mem=100G\n")
        fh.writelines("#SBATCH --output=out/tas_slp_anom_AMJJAS.out.%j\n")
        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L0/tas_slp_anom_AMJJAS_creator.py " + LOC + " > logs/tas_slp_anom_AMJJAS_creator_" + LOC+ ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("sbatch %s" %job_file)


