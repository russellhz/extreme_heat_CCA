#!/usr/bin/env python

import os


LOCS = ["Northeast", "NorthernRockiesandPlains", "Northwest", "OhioValley", "South", "Southeast", "Southwest", "UpperMidwest", "West"]  
N_sets = [[150, 100, 1, 1798], [150, 100, 5, 1798], [150, 100, 10, 1798], [150, 100, 20, 1798]]
os.chdir("/glade/u/home/horowitz/tvar_dyn_adj/scriptrunner/")

for LOC in LOCS:
    for sets in N_sets:
        N = str(sets[0])
        N_S = str(sets[1])
        N_R = str(sets[2])
        N_Y = str(sets[3])
        args = " ".join([LOC, N, N_S, N_R, N_Y])
        args_ = "_".join([LOC, N, N_S, N_R, N_Y])
        job_file = "jobs/" + args_ + '.sbatch'
        open(job_file, 'a')

        with open(job_file, "w") as fh:
            fh.writelines("#!/bin/bash -l\n")
            fh.writelines("#SBATCH --job-name=dyn_adj_sens\n")
            fh.writelines("#SBATCH --account=P04010022\n")
            fh.writelines("#SBATCH --ntasks=1\n")
            fh.writelines("#SBATCH --time=10:00:00\n")
            fh.writelines("#SBATCH --partition=dav\n")
            fh.writelines("#SBATCH --output=out/dyn_adj_xperiment.out.%j\n")
            fh.writelines("module load ncarenv\n")
            fh.writelines("module load python\n")
            fh.writelines("ncar_pylib my_npl_clone_casper\n")
            fh.writelines("python ../py/L0/dyn_adj_sensitivity.py " + args + " > logs/dyn_adj_sensitivity_" + args_ + ".log\n")
            fh.writelines("deactivate\n")
            
        os.system("sbatch %s" %job_file)


