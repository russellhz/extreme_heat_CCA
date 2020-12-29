#!/usr/bin/env python

import os

LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
PICTLS = ['PICTL', 'SSTPICTL']

os.chdir("/glade/u/home/horowitz/tvar_dyn_adj/scriptrunner/")

for LOC in LOCS:
    for PICTL in PICTLS:
        args = " ".join([LOC,PICTL])
        args_ = "_".join([LOC,PICTL])

        job_file = "jobs/distance_matrix_" + args_ + '.sbatch'
        open(job_file, 'a')

        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash -l\n")
            fh.writelines("#SBATCH --job-name=dyn_adj_xperiment\n")
            fh.writelines("#SBATCH --account=P04010022\n")
            fh.writelines("#SBATCH --ntasks=1\n")
            fh.writelines("#SBATCH --time=03:00:00\n")
            fh.writelines("#SBATCH --partition=dav\n")
            fh.writelines("#SBATCH --output=out/dyn_adj_xperiment.out.%j\n")
            fh.writelines("module load ncarenv\n")
            fh.writelines("module load python\n")
            fh.writelines("ncar_pylib my_npl_clone_casper\n")
            fh.writelines("python ../py/L0/distance_matrix.py "  + args + " > ../logs/distance_matrix" + args_ +".log\n")
            fh.writelines("deactivate\n")
            
        os.system("sbatch %s" %job_file)


