#!/usr/bin/env python

import os
import glob
import numpy as np
import re
                
# Open full tas file
CESMLE_DIR = '/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/'
# 20th century
slp_fns = [f for f in glob.glob(CESMLE_DIR + "PSL/b.e11.B20TRC5CNBDRD.f09_g16.[0-9]" + "*.nc")]
# RCP 8.5
slp_fns.extend([f for f in glob.glob(CESMLE_DIR + "PSL/b.e11.BRCP85C5CNBDRD.f09_g16.[0-9]" + "*.nc")])

# Find codes
ensemble_codes = np.unique([re.search(r'(?<=.)[0-9]{3}(?=.)', i).group() for i in slp_fns])
ensemble_codes = np.sort(ensemble_codes)  

for code in ensemble_codes:
    job_file = "jobs/" + 'cesmle_anom_JJA_creator_'+ code +'.sbatch'
    open(job_file, 'a')

    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash -l\n")
        fh.writelines("#SBATCH --job-name=cesmle_anom\n")
        fh.writelines("#SBATCH --account=P04010022\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --time=01:00:00\n")
        fh.writelines("#SBATCH --partition=dav\n")
        fh.writelines("#SBATCH --output=out/cesmle_anom.out.%j\n")
        fh.writelines("module load ncarenv\n")
        fh.writelines("module load python\n")
        fh.writelines("ncar_pylib my_npl_clone_casper\n")
        fh.writelines("python ../py/L0/cesmle_anom_JJA_creator.py " + code + " > logs/cesmle_anom_JJA_creator_" + code + ".log\n")
        fh.writelines("deactivate\n")
        
    os.system("sbatch %s" %job_file)


