#!/usr/bin/env python

import os

os.chdir("/glade/u/home/horowitz/tvar_dyn_adj/scriptrunner/")


job_file = "jobs/" + 'distance_matrix.sbatch'
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
    fh.writelines("python ../py/L0/distance_matrix.py " + " > ../logs/distance_matrix.log\n")
    fh.writelines("deactivate\n")
    
os.system("sbatch %s" %job_file)


