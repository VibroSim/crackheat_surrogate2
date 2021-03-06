# !!!!***** WARNING: OBSOLETE; Not upgraded with many fixes from eval_crackheat_surrogate2.py ***!!!
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

from crackheat_surrogate2.load_surrogate_shear import load_denorm_surrogates_shear_from_jsonfile
from crackheat_surrogate2.training_eval_shear import training_eval_shear

surrogate_eval_nsmap={
    "dc": "http://limatix.org/datacollect",
    "dcv": "http://limatix.org/dcvalue",
    "xlink": "http://www.w3.org/1999/xlink",
}


eval_crackheat_pool = multiprocessing.Pool()
eval_crackheat_threadpool = multiprocessing.dummy.Pool()
#eval_crackheat_pool=None
#eval_crackheat_threadpool=None
eval_crackheat_threadlock = threading.Lock()  # MUST be used when calling multiprocessing functions, matplotlib functions, and lxml functions from thread


def plot_slices(dc_dest_href,
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
                peakidx,
                mu_val,log_msqrtR_val,log_crack_model_shear_factor_val,
                main_thread_todo_list,main_thread_stuff_todo):
    
    output_plots = []

    for axis in range(3):
        # parameters are mu, log_msqrtR, log_crack_model_shear_factor
        
        # Use these constant values except for the given axis
        testgrid_const_vals = np.array((mu_val,log_msqrtR_val,log_crack_model_shear_factor_val),dtype='d')
            
        # Use these values for the given axis
        testgrid_var_vals = np.linspace(min_vals[axis],max_vals[axis],50)
        
        testgrid = np.ones((testgrid_var_vals.shape[0],1),dtype='d')*testgrid_const_vals[np.newaxis,:]
        testgrid[:,axis] = testgrid_var_vals
            
        eval_crackheat_threadlock.acquire() # pandas is documented as thread-unsafe.... this is probably unnecessary...
        try:
            testgrid_dataframe=pd.DataFrame(testgrid,columns=["mu","log_msqrtR","log_crack_model_shear_factor"])
            sur_out = surrogate.evaluate(testgrid_dataframe)
        
            
            testgrid_dict = { "mu": testgrid[:,0],
                              "log_msqrtR": testgrid[:,1],
                              "log_crack_model_shear_factor": testgrid[:,2]
            }
            pass
        finally:
            eval_crackheat_threadlock.release() # pandas is documented as thread-unsafe.... this is probably unnecessary...
            pass
        (direct,direct_stddev) = training_eval_shear(testgrid_dict,surrogate.bendingstress,surrogate.dynamicnormalstressampl,surrogate.dynamicshearstressampl,tortuosity,
                                                     dc_closurestress_side1_href.getpath(), # closure stress left side csv
                                                     dc_closurestress_side2_href.getpath(), # closure stress right side csv
                                                     dc_a_side1_numericunits.value('m'), # crack length left side
                                                     dc_a_side2_numericunits.value('m'), # crack length right side
                                                     sigma_yield,
                                                     tau_yield,
                                                     crack_model_normal_type_str,
                                                     crack_model_shear_type_str,
                                                     E,nu,
                                                     numdraws_int,
                                                     multiprocessing_pool=eval_crackheat_pool,
                                                     multiprocessing_lock=eval_crackheat_threadlock)
        
        def gen_plot():
            eval_crackheat_threadlock.acquire() # matplotlib is thread-unsafe.... this was definitely necessary... but maybe no longer now that execution is delegated back to main thread... surrogate_eval_doc is also thread-unsafe...
            try:
                pl.figure()
                pl.plot(testgrid_var_vals/axisunitfactor[axis],sur_out["mean"],'-',
                        testgrid_var_vals/axisunitfactor[axis],direct,'-',
                        testgrid_var_vals/axisunitfactor[axis],sur_out['lower95'],'--',
                        testgrid_var_vals/axisunitfactor[axis],direct+direct_stddev,':',
                        (testgrid_var_vals[0]/axisunitfactor[axis],testgrid_var_vals[-1]/axisunitfactor[axis]),(surrogate.thermalpower/surrogate.excfreq,surrogate.thermalpower/surrogate.excfreq),'-',
                        testgrid_var_vals/axisunitfactor[axis],sur_out['upper95'],'--',
                        testgrid_var_vals/axisunitfactor[axis],direct-direct_stddev,':')
                pl.grid()
                pl.xlabel('%s (%s)' % (axisnames[axis],axisunits[axis]))
                pl.ylabel('Heating per cycle (Joules)')
                pl.legend(('Surrogate','Direct','Surrogate bounds','Direct bounds','Observation'),loc='best')
                title = ""
                if axis != 0:
                    title += " mu = %f" % (mu_val)
                    pass
                if axis != 1:
                    title += " ln msqrtR = %f ln(sqrt(m)/(m*m))" % (log_msqrtR_val)
                    pass
                if axis != 2:
                    title += " ln shearfact = %f ln(unitless)" % (log_crack_model_shear_factor_val)
                    pass
                
                title += "\nbending=%.1f MPa normal=%.1f MPa shear=%.1f MPa" % (surrogate.bendingstress/1e6,surrogate.dynamicnormalstressampl/1e6,surrogate.dynamicshearstressampl/1e6)
                pl.title(title)
                outputplot_href = hrefv("%s_surrogateeval_%.1fMPa_%.1fMPa_%.1fMPa_%.2d_%.1d.png" % (dc_specimen_str,surrogate.bendingstress/1e6,surrogate.dynamicnormalstressampl/1e6,surrogate.dynamicshearstressampl/1e6,peakidx,axis),contexthref=dc_dest_href)
        
                pl.savefig(outputplot_href.getpath(),dpi=300)
                plot_el = surrogate_eval_doc.addelement(surrogate_eval_doc.getroot(),"dc:surrogateplot")
                outputplot_href.xmlrepr(surrogate_eval_doc,plot_el)

                output_plots.append(outputplot_href)
                pass
            finally:
                eval_crackheat_threadlock.release() # pandas is documented as thread-unsafe.... this is probably unnecessary...
                pass
            pass

        delegate_to_main_thread(main_thread_todo_list,main_thread_stuff_todo,gen_plot)
        pass
        
    return output_plots

def delegate_to_main_thread(main_thread_todo_list,main_thread_stuff_todo,action):
    if action is not None:
        #print("Delegating task: %d" % (id(action)))
        complete_condition = threading.Condition()
        pass
    else:
        #print("Terminating delegation")
        complete_condition = None
        pass

    with main_thread_stuff_todo:
        main_thread_todo_list.append((action,complete_condition))
        main_thread_stuff_todo.notify()
        pass

    if action is not None:
        with complete_condition: # Wait for execution to complete. 
            complete_condition.wait()
            pass
        pass

    pass



def snap_to_gridlines(surrogate,
                      mu_val,
                      log_msqrtR_val,
                      log_crack_model_shear_factor_val):

    param_scaling = np.array((surrogate.params_nominal["mu"],
                              surrogate.params_nominal["log_msqrtR"],
                              1.0), # no scaling for log_crack_model_shear_factor
                             dtype='d')
    trained_vals = surrogate.X*param_scaling[np.newaxis,:]
    trained_mu = np.unique(trained_vals[:,0])
    trained_log_msqrtR = np.unique(trained_vals[:,1])
    trained_log_crack_model_shear_factor = np.unique(trained_vals[:,2])
    
    nearest = lambda trained,val : trained[np.argmin(np.abs(trained-val))]
    
    mu_val = nearest(trained_mu, mu_val)
    log_msqrtR_val = nearest(trained_log_msqrtR,log_msqrtR_val)
    log_crack_model_shear_factor_val = nearest(trained_log_crack_model_shear_factor,log_crack_model_shear_factor_val)
    
    
    return (mu_val,
            log_msqrtR_val,
            log_crack_model_shear_factor_val)


def eval_crackheat_singlesurrogate(params):

    (fixedparams,surrogate_key) = params
    (surrogates,
     surrogate_eval_doc,
     biggrid_expanded,
     biggrid_dataframe,
     only_on_gridlines_bool,
     dc_dest_href,
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
     axisnames,
     axisunits,
     axisunitfactor,
     min_vals,
     max_vals,
     dc_ecs_traces_per_data_point_float,
     main_thread_todo_list,
     main_thread_stuff_todo) = fixedparams

    #print("eval_crackheat_singlesurrogate")
    
    output_plots = []
    bigsur_out = surrogates[surrogate_key].evaluate(biggrid_dataframe)
    
    # find peaks in sd
    sd_expanded = bigsur_out["sd"].reshape(biggrid_expanded[0].shape)

    sd_peaks = ( (sd_expanded[1:-1,1:-1] > sd_expanded[0:-2,1:-1]) &
                 (sd_expanded[1:-1,1:-1] > sd_expanded[2:,1:-1]) &
                 (sd_expanded[1:-1,1:-1] > sd_expanded[1:-1,0:-2]) &
                 (sd_expanded[1:-1,1:-1] > sd_expanded[1:-1,2:]))

    # find peaks in mean
    
    mean_expanded = bigsur_out["mean"].reshape(biggrid_expanded[0].shape)
    
    mean_peaks = ( (mean_expanded[1:-1,1:-1] > mean_expanded[0:-2,1:-1]) &
                   (mean_expanded[1:-1,1:-1] > mean_expanded[2:,1:-1]) &
                   (mean_expanded[1:-1,1:-1] > mean_expanded[1:-1,0:-2]) &
                   (mean_expanded[1:-1,1:-1] > mean_expanded[1:-1,2:]))
    
    #sd_peaklocs = np.where(sd_peaks)
    
    sd_peakvals = sd_expanded[1:-1,1:-1][sd_peaks]
    sd_peak_mu = biggrid_expanded[0][1:-1,1:-1][sd_peaks]
    sd_peak_log_msqrtR = biggrid_expanded[1][1:-1,1:-1][sd_peaks]
    sd_peak_log_crack_model_shear_factor = biggrid_expanded[2][1:-1,1:-1][sd_peaks]
    
    sd_peaksort = np.argsort(sd_peakvals)

    mean_peakvals = mean_expanded[1:-1,1:-1][mean_peaks]
    mean_peak_mu = biggrid_expanded[0][1:-1,1:-1][mean_peaks]
    mean_peak_log_msqrtR = biggrid_expanded[1][1:-1,1:-1][mean_peaks]
    mean_peak_log_crack_model_shear_factor = biggrid_expanded[2][1:-1,1:-1][mean_peaks]
    
    mean_peaksort = np.argsort(mean_peakvals)

    num_peaks_per_data_point = dc_ecs_traces_per_data_point_float/3.0 # 3.0 is number of axes

    max_peaks_per_data_point=int(np.ceil(num_peaks_per_data_point))
    
    peakidx=-1
    for peakidx in range(min(max_peaks_per_data_point,mean_peakvals.shape[0])): # Use up to 2 peaks corresponding to relative maxima

        rand_val = np.random.rand()
        if rand_val > (num_peaks_per_data_point/max_peaks_per_data_point):
            continue # skip this point
        #mu_val = sd_peak_mu[sd_peaksort][-peakidx-1]
        #log_msqrtR_val = sd_peak_log_msqrtR[sd_peaksort][-peakidx-1]

        mu_val = mean_peak_mu[mean_peaksort][-peakidx-1]
        log_msqrtR_val = mean_peak_log_msqrtR[mean_peaksort][-peakidx-1]
        log_crack_model_shear_factor_val = mean_peak_log_crack_model_shear_factor[mean_peaksort][-peakidx-1]
        
        #raise ValueError("debug!")
        if only_on_gridlines_bool:
            (mu_val,
             log_msqrtR_val,
             log_crack_model_shear_factor_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                                   mu_val,
                                                                   log_msqrtR_val,
                                                                   log_crack_model_shear_factor_val)
                
            pass
            
        output_plots.extend(plot_slices(dc_dest_href,
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
                                        surrogates[surrogate_key],
                                        axisnames,
                                        axisunits,
                                        axisunitfactor,
                                        min_vals,
                                        max_vals,
                                        peakidx,
                                        mu_val,log_msqrtR_val,log_crack_model_shear_factor_val,
                                        main_thread_todo_list,
                                        main_thread_stuff_todo))
        pass
        
    peakidx += 1
    
    rand_val = np.random.rand()
    if peakidx < max_peaks_per_data_point and rand_val <= (num_peaks_per_data_point/max_peaks_per_data_point):
        # Did not display 2 peaks corresponding to relative maxima... Use peak corresponding to absolute maximum as well
        #idx_absmax = np.argmax(sd_expanded)
        #idxs_absmax = np.unravel_index(idx_absmax,sd_expanded.shape)

        idx_absmax = np.argmax(mean_expanded)
        idxs_absmax = np.unravel_index(idx_absmax,mean_expanded.shape)
        
        mu_val = biggrid_expanded[0][idxs_absmax]
        log_msqrtR_val = biggrid_expanded[1][idxs_absmax]
        log_crack_model_shear_factor_val = biggrid_expanded[2][idxs_absmax]
        

        if only_on_gridlines_bool:
            (mu_val,
             log_msqrtR_val,
             log_crack_model_shear_factor_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                                   mu_val,
                                                                   log_msqrtR_val,
                                                                   log_crack_model_shear_factor_val)
            pass
            
        output_plots.extend(plot_slices(dc_dest_href,
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
                                        surrogates[surrogate_key],
                                        axisnames,
                                        axisunits,
                                        axisunitfactor,
                                        min_vals,
                                        max_vals,
                                        peakidx,
                                        mu_val,log_msqrtR_val,log_crack_model_shear_factor_val,
                                        main_thread_todo_list,
                                        main_thread_stuff_todo))
        
        pass
        
    return output_plots
        

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
        dc_shear_surrogate_href,
        dc_sheartraining_min_mu_float,
        dc_sheartraining_max_mu_float,
        dc_sheartraining_min_log_msqrtR_float,
        dc_sheartraining_max_log_msqrtR_float,
        dc_ecs_traces_per_data_point_float=1.0,
        numdraws_int = 10,
        only_on_gridlines_bool = False,
        crack_model_normal_type_str = "Tada_ModeI_CircularCrack_along_midline",
        crack_model_shear_type_str = "Fabrikant_ModeII_CircularCrack_along_midline"):

        
    
    sigma_yield = dc_spcYieldStrength_numericunits.value("Pa")
    tau_yield=sigma_yield/2.0
    
    E=dc_spcYoungsModulus_numericunits.value("Pa")
    nu = dc_spcPoissonsRatio_numericunits.value("unitless")   #Poisson's Ratio


    tortuosity = dc_filtered_sigma_numericunits.value('radians')*180.0/np.pi # training_eval() wants tortuosity in degrees!

    # xml sub-document to hold plot output hrefs
    surrogate_eval_doc = xmldoc.newdoc("dc:surrogate_eval",nsmap=surrogate_eval_nsmap,contexthref=_xmldoc.getcontexthref())
    
    # Load in surrogates
    surrogates = load_denorm_surrogates_shear_from_jsonfile(dc_shear_surrogate_href.getpath(),nonneg=True)

    axisnames = ['mu','log msqrtR','log shearfact']
    axisunits = ['unitless','ln(sqrt(m)/(m*m))','Unitless']
    axisunitfactor = [ 1.0, 1.0 ,1.0 ]
    
    # biggrid .. we go through biggrid in the Surrogate finding
    # the worst case standard deviations. Then we use these as the
    # center points for cuts along each axis plotting the
    # surrogate, raw data, and uncertainty

    # !!!*** NOTE: the bounds of the seq's in TrainSurrogate_shear.R
    # should match min_vals and max_vals!!!***
    min_vals = np.array((dc_sheartraining_min_mu_float,dc_sheartraining_min_log_msqrtR_float,-2.0),dtype='d')
    max_vals = np.array((dc_sheartraining_max_mu_float,dc_sheartraining_max_log_msqrtR_float,6.0),dtype='d')

    # rough equivalent of R expand.grid():
    biggrid_expanded = np.meshgrid(
        np.linspace(min_vals[0],max_vals[0],4), # mu
        np.linspace(min_vals[1],max_vals[1],4), # log_msqrtR
        np.linspace(min_vals[2],max_vals[2],4), # log_crack_model_shear_factor
    )
    
    biggrid = np.stack(biggrid_expanded,-1).reshape(-1,3)
    
    biggrid_dataframe=pd.DataFrame(biggrid,columns=["mu","log_msqrtR","log_crack_model_shear_factor"])

    #raise ValueError("debug")

    main_thread_todo_list=[]
    main_thread_stuff_todo=threading.Condition()

    fixedparams=(surrogates,
                 surrogate_eval_doc,
                 biggrid_expanded,
                 biggrid_dataframe,
                 only_on_gridlines_bool,
                 dc_dest_href,
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
                 axisnames,
                 axisunits,
                 axisunitfactor,
                 min_vals,
                 max_vals,
                 dc_ecs_traces_per_data_point_float,
                 main_thread_todo_list,
                 main_thread_stuff_todo)
    
    paramlist = []

    for surrogate_key in surrogates:
        params = (fixedparams,surrogate_key)
        paramlist.append(params)
        pass

    


    if eval_crackheat_threadpool is None:
        resultlist = list(map(eval_crackheat_singlesurrogate,paramlist))
        pass
    else:
        # We are running multi-threaded
        mapthread_result_container=[]
        #print("Spawning mapthread")
        def mapthread_code():
            generated_resultlist = list(eval_crackheat_threadpool.map(eval_crackheat_singlesurrogate,paramlist)) # updates surrogate_eval_doc using proper locking

            mapthread_result_container.append(generated_resultlist)

            #print("generated_resultlist=%s" % (str(generated_resultlist)))
            delegate_to_main_thread(main_thread_todo_list,main_thread_stuff_todo,None) # None is an indicator that everything is complete
            pass

        mapthread = threading.Thread(target = mapthread_code) 
        mapthread.start()
        
        #print("Working on delegated tasks")

        # Do stuff delegated to us by threads
        while True:
            with main_thread_stuff_todo:

                # Wait for something available. 
                while len(main_thread_todo_list)==0:
                    main_thread_stuff_todo.wait()
                    pass
                    
                (todo,complete_condition) = main_thread_todo_list.pop()
                #print("Got delegated task: %d" % (id(todo)))
                if todo is None: # None is a flag indicating all is complete
                    #print("All tasks complete")
                    break
                todo()  # Call function with stuff to do  in main thread (i.e. matplotlib plotting)
                with complete_condition:
                    complete_condition.notify() # Notify that we have completed this code. 
                    pass
                pass
            pass
        # Wait for mapthread to exit
        mapthread.join()
        resultlist = mapthread_result_container[0]
        pass
        


    return {
        "dc:surrogate_eval": xmltreev(surrogate_eval_doc)
    }

