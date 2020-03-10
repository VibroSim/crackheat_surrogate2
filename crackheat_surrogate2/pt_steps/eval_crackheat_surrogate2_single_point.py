import sys
import os
import os.path
import csv
import ast
import copy
import posixpath
import subprocess
import threading
import multiprocessing
import multiprocessing.dummy # Gives thread pools not process pools
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import xmltreevalue as xmltreev
from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.xmldoc import xmldoc

from crackheat_surrogate2.load_surrogate import load_denorm_surrogates_from_jsonfile
from crackheat_surrogate2.eval_surrogate import plot_slices

surrogate_eval_nsmap={
    "dc": "http://limatix.org/datacollect",
    "dcv": "http://limatix.org/dcvalue",
    "xlink": "http://www.w3.org/1999/xlink",
}


eval_crackheat_pool = multiprocessing.Pool()
#eval_crackheat_pool=None
eval_crackheat_threadlock = threading.Lock()  # MUST be used when calling multiprocessing functions, matplotlib functions, and lxml functions from thread



def delegate_to_main_thread(main_thread_todo_list,main_thread_stuff_todo,action):
    action()
    pass



def run(_xmldoc,_element,
        dc_dest_href,
        _inputfilename,
        dc_specimen_str,
        dc_spcmaterial_str,
        dc_filtered_sigma_numericunits, #=numericunitsv(30.0*np.pi/180.0,"radians"), # filtered tortuosity sigma in radians
        dc_closurestress_side1_href, # closure stress left side csv
        dc_closurestress_side2_href, # closure stress right side csv
        dc_a_side1_numericunits, # crack length left side
        dc_a_side2_numericunits, # crack length right side
        dc_spcYieldStrength_numericunits,
        dc_spcYoungsModulus_numericunits,
        dc_spcPoissonsRatio_numericunits,
        dc_surrogate_href,
        dc_ecs2sp_mu_val_numericunits,
        dc_ecs2sp_msqrtR_val_numericunits,
        dc_ecs2sp_measnum_int,
        dc_crackheat_table_href,
        dc_ecs_traces_per_data_point_float=1.0,
        numdraws_int = 10,
        only_on_gridlines_bool = False,
        crack_model_normal_type_str = "Tada_ModeI_CircularCrack_along_midline",
        crack_model_shear_type_str = "Fabrikant_ModeII_CircularCrack_along_midline"):


    crackheatfile_dataframe = pd.read_csv(dc_crackheat_table_href.getpath(),dtype={"DynamicNormalStressAmpl (Pa)": str, "DynamicShearStressAmpl (Pa)": str, "BendingStress (Pa)": str})
    lookup_keys = "bs_pa_" + crackheatfile_dataframe["BendingStress (Pa)"] + "_dnsa_pa_" + crackheatfile_dataframe["DynamicNormalStressAmpl (Pa)"] + "_dssa_pa_" + crackheatfile_dataframe["DynamicShearStressAmpl (Pa)"]
    key_idx = np.where(crackheatfile_dataframe["measnum"].astype(np.int32) == dc_ecs2sp_measnum_int)[0][0]
    key = lookup_keys[key_idx]
    

    sigma_yield = dc_spcYieldStrength_numericunits.value("Pa")
    tau_yield=sigma_yield/2.0
    
    E=dc_spcYoungsModulus_numericunits.value("Pa")
    nu = dc_spcPoissonsRatio_numericunits.value("unitless")   #Poisson's Ratio


    tortuosity = dc_filtered_sigma_numericunits.value('radians')*180.0/np.pi # training_eval() wants tortuosity in degrees!

    # xml sub-document to hold plot output hrefs
    surrogate_eval_doc = xmldoc.newdoc("dc:surrogate_eval_single_point",nsmap=surrogate_eval_nsmap,contexthref=_xmldoc.getcontexthref())
    
    # Load in surrogates
    surrogates = load_denorm_surrogates_from_jsonfile(dc_surrogate_href.getpath(),nonneg=True)

    surrogate = surrogates[key]

    axisnames = ['log mu','log msqrtR']
    axisunits = ['unitless','ln(sqrt(m)/(m*m))']
    axisunitfactor = [ 1.0, 1.0 ]
    
    # biggrid .. we go through biggrid in the Surrogate finding
    # the worst case standard deviations. Then we use these as the
    # center points for cuts along each axis plotting the
    # surrogate, raw data, and uncertainty

    # !!!*** NOTE: the bounds of the seq's in TrainSurrogate.R
    # should match min_vals and max_vals!!!***
    min_vals = np.array((np.log(0.01),9.2),dtype='d')
    max_vals = np.array((np.log(2.0),18.4),dtype='d')

    # rough equivalent of R expand.grid():
    biggrid_expanded = np.meshgrid(
        np.linspace(min_vals[0],max_vals[0],7), # log_mu
        np.linspace(min_vals[1],max_vals[1],8)) # log_msqrtR

    biggrid = np.stack(biggrid_expanded,-1).reshape(-1,2)
    
    biggrid_dataframe=pd.DataFrame(biggrid,columns=["log_mu","log_msqrtR"])

    #raise ValueError("debug")

    main_thread_todo_list=[]
    main_thread_stuff_todo=threading.Condition()

    output_plots = plot_slices(dc_dest_href,
                               dc_specimen_str,
                               dc_closurestress_side1_href,
                               dc_closurestress_side2_href,
                               dc_a_side1_numericunits,
                               dc_a_side2_numericunits,
                               numdraws_int,
                               crack_model_normal_type_str,
                               crack_model_shear_type_str,
                               sigma_yield,
                               tau_yield,
                               E,nu,
                               tortuosity,
                               surrogate_eval_doc,
                               surrogate,
                               axisnames,
                               axisunits,
                               axisunitfactor,
                               min_vals,
                               max_vals,
                               -1,
                               np.log(dc_ecs2sp_mu_val_numericunits.value()), # log_mu_val
                               np.log(dc_ecs2sp_msqrtR_val_numericunits.value()), # log_msqrtR_val,
                               main_thread_todo_list,
                               main_thread_stuff_todo,
                               eval_crackheat_pool,
                               eval_crackheat_threadlock,
                               delegate_to_main_thread)
    



    return {
        "dc:surrogate_eval": xmltreev(surrogate_eval_doc)
    }

