import sys
import csv
import ast
import copy
import posixpath
import numpy as np
import tempfile
import os
import os.path
import subprocess

from ..paths import get_rscripts_path

def main(args=None):
    if args is None:
        args=sys.argv
        pass    


    if len(args) != 15:
        print("Usage: train_crackheat_surrogate2 <tortuosity_degrees> <leftclosure_csv> <rightclosure_csv> <crackheat_csv> <closure_lowest_avg_load_used_Pa> <aleft_m> <aright_m> <sigma_yield_Pa> <tau_yield_Pa> <crack_model_normal_type> <crack_model_shear_type> <E_Pa> <PoissonsRatio> <numdraws> <num_mus> <num_msqrtRs> <output_filename_json>")
        sys.exit(0)
        pass

    # Transfer params verbatim, but adding python binary as final parameter
    params=args[1:17]
    
    python_binary = sys.executable
    params.append(python_binary)

    Rscript = os.path.join(get_rscripts_path(),"TrainSurrogate.R")
    
    # print parameters for pasting into R
    print("R parameter specification for %s:" % (Rscript))
    print("args = c(")
    for param in params[:-1]:
        print("         \"%s\"," % (param)) # note: should really preprocess param to do proper escaping of quote characters, etc.
        pass
    print("         \"%s\")" % (params[-1])) # note: should really preprocess param to do proper escaping of quote characters, etc.

    
    subprocess.check_call([ "Rscript", "--slave", "--no-save", "--no-restore", Rscript ] + params)
    
    sys.exit(0)
    pass
