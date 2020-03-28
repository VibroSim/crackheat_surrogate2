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


from crackheat_surrogate2.training_eval import training_eval

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
                log_mu_val,log_msqrtR_val,
                main_thread_todo_list,
                main_thread_stuff_todo,
                eval_crackheat_pool,
                eval_crackheat_threadlock,
                delegate_to_main_thread):

    output_plots = []
    
    for axis in range(2):
        # parameters are log_mu, log_msqrtR
        
        # Use these constant values except for the given axis
        testgrid_const_vals = np.array((log_mu_val,log_msqrtR_val),dtype='d')
            
        # Use these values for the given axis
        testgrid_var_vals = np.linspace(min_vals[axis],max_vals[axis],30)
        
        testgrid = np.ones((testgrid_var_vals.shape[0],1),dtype='d')*testgrid_const_vals[np.newaxis,:]
        testgrid[:,axis] = testgrid_var_vals
            
        eval_crackheat_threadlock.acquire() # pandas is documented as thread-unsafe.... this is probably unnecessary...
        try:
            testgrid_dataframe=pd.DataFrame(testgrid,columns=["log_mu","log_msqrtR"])
            sur_out = surrogate.evaluate(testgrid_dataframe)
        
            
            testgrid_dict = { "log_mu": testgrid[:,0],
                              "log_msqrtR": testgrid[:,1] }
            pass
        finally:
            eval_crackheat_threadlock.release() # pandas is documented as thread-unsafe.... this is probably unnecessary...
            pass
        ( direct,
          direct_stddev,
          soft_closure_diags
        ) = training_eval(testgrid_dict,surrogate.bendingstress,surrogate.dynamicnormalstressampl,surrogate.dynamicshearstressampl,tortuosity,
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
                    title += " ln mu = %f" % (log_mu_val)
                    pass
                if axis != 1:
                    title += " ln msqrtR = %f ln(sqrt(m)/(m*m))" % (log_msqrtR_val)
                    pass
                
                title += "\nbending=%.1f MPa normal=%.1f MPa shear=%.1f MPa" % (surrogate.bendingstress/1e6,surrogate.dynamicnormalstressampl/1e6,surrogate.dynamicshearstressampl/1e6)
                pl.title(title)
                outputplot_href = hrefv("%s_surrogateeval_%.1fMPa_%.1fMPa_%.1fMPa_%.2d_%.1d.png" % (dc_specimen_str,surrogate.bendingstress/1e6,surrogate.dynamicnormalstressampl/1e6,surrogate.dynamicshearstressampl/1e6,peakidx,axis),contexthref=dc_dest_href)
                
                pl.savefig(outputplot_href.getpath(),dpi=300)
                plot_el = surrogate_eval_doc.addelement(surrogate_eval_doc.getroot(),"dc:surrogateplot")
                outputplot_href.xmlrepr(surrogate_eval_doc,plot_el)
                
                output_plots.append(outputplot_href)

                # soft closure diag plot
                pl.figure()
                pl.plot(np.arange(soft_closure_diags.shape[0]),soft_closure_diags[:,0],'-',
                        np.arange(soft_closure_diags.shape[0]),soft_closure_diags[:,1],'-',
                        np.arange(soft_closure_diags.shape[0]),soft_closure_diags[:,2],'-',
                        np.arange(soft_closure_diags.shape[0]),soft_closure_diags[:,3],'-')
                pl.grid(True)
                pl.xlabel('SC calculation index')
                pl.ylabel('Residual')
                pl.legend(('residual_sub_left','residual_add_left','residual_sub_right','residual_add_right'))
                pl.title('title')


                scdiag_outputplot_href = hrefv("%s_surrogateeval_%.1fMPa_%.1fMPa_%.1fMPa_%.2d_%.1d_scdiag.png" % (dc_specimen_str,surrogate.bendingstress/1e6,surrogate.dynamicnormalstressampl/1e6,surrogate.dynamicshearstressampl/1e6,peakidx,axis),contexthref=dc_dest_href)
                
                pl.savefig(scdiag_outputplot_href.getpath(),dpi=300)
                scdiag_plot_el = surrogate_eval_doc.addelement(surrogate_eval_doc.getroot(),"dc:surrogateplot_scdiag")
                scdiag_outputplot_href.xmlrepr(surrogate_eval_doc,scdiag_plot_el)
                
                output_plots.append(scdiag_outputplot_href)


                pass
            finally:
                eval_crackheat_threadlock.release() # pandas is documented as thread-unsafe.... this is probably unnecessary...
                pass
            pass

        delegate_to_main_thread(main_thread_todo_list,main_thread_stuff_todo,gen_plot)
        pass
        
    return output_plots



def snap_to_gridlines(surrogate,
                      log_mu_val,
                      log_msqrtR_val):

    param_scaling = np.array((surrogate.params_nominal["log_mu"],
                              surrogate.params_nominal["log_msqrtR"]),dtype='d')
    trained_vals = surrogate.X*param_scaling[np.newaxis,:]
    trained_log_mu = np.unique(trained_vals[:,0])
    trained_log_msqrtR = np.unique(trained_vals[:,1])
    
    nearest = lambda trained,val : trained[np.argmin(np.abs(trained-val))]
    
    log_mu_val = nearest(trained_log_mu, log_mu_val)
    log_msqrtR_val = nearest(trained_log_msqrtR,log_msqrtR_val)
    
    
    return (log_mu_val,
            log_msqrtR_val)



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
     main_thread_stuff_todo,
     eval_crackheat_pool,
     eval_crackheat_threadlock,
     delegate_to_main_thread) = fixedparams

    #print("eval_crackheat_singlesurrogate")

    try:
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
        sd_peak_log_mu = biggrid_expanded[0][1:-1,1:-1][sd_peaks]
        sd_peak_log_msqrtR = biggrid_expanded[1][1:-1,1:-1][sd_peaks]
        
        sd_peaksort = np.argsort(sd_peakvals)
        
        mean_peakvals = mean_expanded[1:-1,1:-1][mean_peaks]
        mean_peak_log_mu = biggrid_expanded[0][1:-1,1:-1][mean_peaks]
        mean_peak_log_msqrtR = biggrid_expanded[1][1:-1,1:-1][mean_peaks]
        
        mean_peaksort = np.argsort(mean_peakvals)
        
        num_peaks_per_data_point = dc_ecs_traces_per_data_point_float/2.0 # 2.0 is number of axes
        
        max_peaks_per_data_point=int(np.ceil(num_peaks_per_data_point))
    
        peakidx=-1
        for peakidx in range(min(max_peaks_per_data_point,mean_peakvals.shape[0])): # Use up to 2 peaks corresponding to relative maxima
            
            rand_val = np.random.rand()
            if rand_val > (num_peaks_per_data_point/max_peaks_per_data_point):
                continue # skip this point
            #log_mu_val = sd_peak_log_mu[sd_peaksort][-peakidx-1]
            #log_msqrtR_val = sd_peak_log_msqrtR[sd_peaksort][-peakidx-1]

            log_mu_val = mean_peak_log_mu[mean_peaksort][-peakidx-1]
            log_msqrtR_val = mean_peak_log_msqrtR[mean_peaksort][-peakidx-1]
        
            #raise ValueError("debug!")
            if only_on_gridlines_bool:
                (log_mu_val,
                 log_msqrtR_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                     log_mu_val,
                                                     log_msqrtR_val)
                
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
                                            log_mu_val,log_msqrtR_val,
                                            main_thread_todo_list,
                                            main_thread_stuff_todo,
                                            eval_crackheat_pool,
                                            eval_crackheat_threadlock,
                                            delegate_to_main_thread))
            pass
            
        peakidx += 1
    
        rand_val = np.random.rand()
        if peakidx < max_peaks_per_data_point and rand_val <= (num_peaks_per_data_point/max_peaks_per_data_point):
            # Did not display 2 peaks corresponding to relative maxima... Use peak corresponding to absolute maximum as well
            #idx_absmax = np.argmax(sd_expanded)
            #idxs_absmax = np.unravel_index(idx_absmax,sd_expanded.shape)
            
            idx_absmax = np.argmax(mean_expanded)
            idxs_absmax = np.unravel_index(idx_absmax,mean_expanded.shape)
            
            log_mu_val = biggrid_expanded[0][idxs_absmax]
            log_msqrtR_val = biggrid_expanded[1][idxs_absmax]
            
            
            if only_on_gridlines_bool:
                (log_mu_val,
                 log_msqrtR_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                     log_mu_val,
                                                     log_msqrtR_val)
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
                                            log_mu_val,log_msqrtR_val,
                                            main_thread_todo_list,
                                            main_thread_stuff_todo,
                                            eval_crackheat_pool,
                                            eval_crackheat_threadlock,
                                            delegate_to_main_thread))
            
            pass
        pass
    except:
        import traceback
        traceback.print_exc()
        raise
    return output_plots
        
