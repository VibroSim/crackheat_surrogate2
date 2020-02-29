import sys
import os
import os.path
import csv
import ast
import copy
import posixpath
import subprocess
import numpy as np

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import numericunitsvalue as numericunitsv

from crackheat_surrogate2 import get_rscripts_path

def run(_xmldoc,_element,
        dc_dest_href,
        _inputfilename,
        dc_specimen_str,
        dc_spcmaterial_str,
        dc_closurestress_side1_href, # closure stress left side csv
        dc_closurestress_side2_href, # closure stress right side csv
        dc_closure_lowest_avg_load_used_numericunits,
        dc_a_side1_numericunits, # crack length left side
        dc_a_side2_numericunits, # crack length right side
        dc_spcYieldStrength_numericunits,
        dc_spcYoungsModulus_numericunits,
        dc_spcPoissonsRatio_numericunits,
        dc_filtered_sigma_numericunits,# =numericunitsv(30.0*np.pi/180.0,"radians"), # filtered tortuosity sigma in radians
        dc_crackheat_table_href,
        dc_sheartraining_min_mu_float,
        dc_sheartraining_max_mu_float,
        dc_sheartraining_min_log_msqrtR_float,
        dc_sheartraining_max_log_msqrtR_float,
        numdraws_int = 10,
        sheartraining_num_mus_int = 7,
        sheartraining_num_msqrtRs_int = 5,
        sheartraining_num_shearfactors_int = 12,
        crack_model_normal_type_str = "Tada_ModeI_CircularCrack_along_midline",
        crack_model_shear_type_str = "Fabrikant_ModeII_CircularCrack_along_midline"):

    
    # load in crackheat table so that we can precalculate separate surrogates for each tuple of (normal stress, shear stress, bending stress) values
    #crackheatfile_dataframe = pd.read_csv(dc_crackheat_table_href.getpath(),dtype={"DynamicNormalStressAmpl (Pa)": str, "DynamicShearStressAmpl (Pa)": str, "BendingStress (Pa)": str})

    #lookup_keys = "bs_pa_" + crackheatfile_dataframe["BendingStress (Pa)"] + "_dnsa_pa_" + crackheatfile_dataframe["DynamicNormalStressAmpl (Pa)"] + "_dssa_pa_" + crackheatfile_dataframe["DynamicShearStressAmpl (Pa)"]

    #DynamicNormalStressAmpl = crackheatfile_dataframe["DynamicNormalStressAmpl (Pa)"].map(float)
    #DynamicShearStressAmpl = crackheatfile_dataframe["DynamicShearStressAmpl (Pa)"].map(float)
    #BendingStress = crackheatfile_dataframe["BendingStress (Pa)"].map(float)
    
    
    sigma_yield = dc_spcYieldStrength_numericunits.value("Pa")
    tau_yield=sigma_yield/2.0
    
    E=dc_spcYoungsModulus_numericunits.value("Pa")
    nu = dc_spcPoissonsRatio_numericunits.value("unitless")   #Poisson's Ratio

    outputfile_href = hrefv("%s_shear_surrogates.json" % (dc_specimen_str),contexthref=dc_dest_href)

    
        
    TrainSurrogate_params = [ str(dc_filtered_sigma_numericunits.value("radians")*180.0/np.pi),
                              dc_closurestress_side1_href.getpath(),
                              dc_closurestress_side2_href.getpath(),
                              dc_crackheat_table_href.getpath(),
                              str(dc_closure_lowest_avg_load_used_numericunits.value('Pa')),
                              str(dc_a_side1_numericunits.value('m')), # crack length left side
                              str(dc_a_side2_numericunits.value('m')), # crack length right side
                              str(sigma_yield),
                              str(tau_yield),
                              crack_model_normal_type_str,
                              crack_model_shear_type_str,
                              str(E),
                              str(nu),
                              str(numdraws_int),
                              str(dc_sheartraining_min_mu_float),
                              str(dc_sheartraining_max_mu_float),
                              str(sheartraining_num_mus_int),
                              str(dc_sheartraining_min_log_msqrtR_float),
                              str(dc_sheartraining_max_log_msqrtR_float),
                              str(sheartraining_num_msqrtRs_int),
                              str(sheartraining_num_shearfactors_int),
                              outputfile_href.getpath(),
                              sys.executable ]
    
    
    Rscript = os.path.join(get_rscripts_path(),"TrainSurrogate_shear.R")

    # print parameters for pasting into R
    print("R parameter specification for %s:" % (Rscript))
    print("args = c(")
    for param in TrainSurrogate_params[:-1]:
        print("         \"%s\"," % (param)) # note: should really preprocess param to do proper escaping of quote characters, etc.
        pass
    print("         \"%s\")" % (TrainSurrogate_params[-1])) # note: should really preprocess param to do proper escaping of quote characters, etc.
    
    
    subprocess.check_call([ "Rscript", "--slave", "--no-save", "--no-restore", Rscript ] + TrainSurrogate_params)

    return {
        "dc:shear_surrogate": outputfile_href,
        }

