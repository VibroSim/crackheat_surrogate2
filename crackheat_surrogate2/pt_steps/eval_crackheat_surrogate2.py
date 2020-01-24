import sys
import os
import os.path
import csv
import ast
import copy
import posixpath
import subprocess
import multiprocessing
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as pl

from limatix.dc_value import hrefvalue as hrefv
from limatix.dc_value import xmltreevalue as xmltreev
from limatix.dc_value import numericunitsvalue as numericunitsv
from limatix.xmldoc import xmldoc

from crackheat_surrogate2.load_surrogate import load_denorm_surrogates_from_jsonfile
from crackheat_surrogate2.training_eval import training_eval

surrogate_eval_nsmap={
    "dc": "http://limatix.org/datacollect",
    "dcv": "http://limatix.org/dcvalue",
    "xlink": "http://www.w3.org/1999/xlink",
}


eval_crackheat_pool = multiprocessing.Pool()


def plot_slices(_dest_href,
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
                mu_val,log_msqrtR_val):
    
    for axis in range(2):
        # parameters are mu, log_msqrtR
        
        # Use these constant values except for the given axis
        testgrid_const_vals = np.array((mu_val,log_msqrtR_val),dtype='d')
            
        # Use these values for the given axis
        testgrid_var_vals = np.linspace(min_vals[axis],max_vals[axis],50)
        
        testgrid = np.ones((testgrid_var_vals.shape[0],1),dtype='d')*testgrid_const_vals[np.newaxis,:]
        testgrid[:,axis] = testgrid_var_vals
            
        
        testgrid_dataframe=pd.DataFrame(testgrid,columns=["mu","log_msqrtR"])
        sur_out = surrogate.evaluate(testgrid_dataframe)
        
        
        testgrid_dict = { "mu": testgrid[:,0],
                          "log_msqrtR": testgrid[:,1] }
        (direct,direct_stddev) = training_eval(testgrid_dict,surrogate.bendingstress,surrogate.dynamicnormalstressampl,surrogate.dynamicshearstressampl,tortuosity,
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
                                               multiprocessing_pool=eval_crackheat_pool)
        
        pl.figure()
        pl.plot(testgrid_var_vals/axisunitfactor[axis],sur_out["mean"],'-',
                testgrid_var_vals/axisunitfactor[axis],direct,'-',
                testgrid_var_vals/axisunitfactor[axis],sur_out['lower95'],'--',
                testgrid_var_vals/axisunitfactor[axis],direct+direct_stddev,':',
                testgrid_var_vals/axisunitfactor[axis],sur_out['upper95'],'--',
                testgrid_var_vals/axisunitfactor[axis],direct-direct_stddev,':')
        pl.grid()
        pl.xlabel('%s (%s)' % (axisnames[axis],axisunits[axis]))
        pl.legend(('Surrogate','Direct','Surrogate bounds','Direct bounds'),loc='best')
        title = ""
        if axis != 0:
            title += " mu = %f" % (mu_val)
            pass
        if axis != 1:
            title += " ln msqrtR = %f ln(sqrt(m)/(m*m))" % (log_msqrtR_val)
            pass
        pl.title(title)
        outputplot_href = hrefv("%s_surrogateeval_%.2d_%fMPa_%fMPa_%fMPa_%.2d.png" % (dc_specimen_str,bendingstress/1e6,dynamicnormalstressampl/1e6,dynamicshearstressampl/1e6,peakidx,axis),contexthref=_dest_href)
        
        pl.savefig(outputplot_href.getpath(),dpi=300)
        plot_el = surrogate_eval_doc.addelement(surrogate_eval_doc.getroot(),"dc:surrogateplot")
        outputplot_href.xmlrepr(surrogate_eval_doc,plot_el)
        pass
    pass



def snap_to_gridlines(surrogate,
                      mu_val,
                      bendingstress_val,
                      dynamicstress_val,
                      log_msqrtR_val):

    param_scaling = np.array((surrogate.params_nominal["mu"],
                              surrogate.params_nominal["bendingstress"],
                              surrogate.params_nominal["dynamicstress"],
                              surrogate.params_nominal["log_msqrtR"]),dtype='d')
    trained_vals = surrogate.X*param_scaling[np.newaxis,:]
    trained_mu = np.unique(trained_vals[:,0])
    trained_bendingstress = np.unique(trained_vals[:,1])
    trained_dynamicstress = np.unique(trained_vals[:,2])
    trained_log_msqrtR = np.unique(trained_vals[:,3])
    
    nearest = lambda trained,val : trained[np.argmin(np.abs(trained-val))]
    
    mu_val = nearest(trained_mu, mu_val)
    bendingstress_val = nearest(trained_bendingstress, bendingstress_val)
    dynamicstress_val = nearest(trained_dynamicstress, dynamicstress_val)
    log_msqrtR_val = nearest(trained_log_msqrtR,log_msqrtR_val)
    
    
    return (mu_val,
            bendingstress_val,
            dynamicstress_val,
            log_msqrtR_val)
   

def run(_xmldoc,_element,
        _dest_href,
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
    surrogates = load_denorm_surrogates_from_jsonfile(dc_surrogate_href.getpath(),nonneg=True)

    axisnames = ['mu','log msqrtR']
    axisunits = ['unitless','ln(sqrt(m)/(m*m))']
    axisunitfactor = [ 1.0, 1.0 ]
    
    # biggrid .. we go through biggrid in the Surrogate finding
    # the worst case standard deviations. Then we use these as the
    # center points for cuts along each axis plotting the
    # surrogate, raw data, and uncertainty

    # !!!*** NOTE: the bounds of the seq's in TrainSurrogate.R
    # should match min_vals and max_vals!!!***
    min_vals = np.array((0.02,9.2),dtype='d')
    max_vals = np.array((2.0,18.4),dtype='d')

    # rough equivalent of R expand.grid():
    biggrid_expanded = np.meshgrid(
        np.linspace(min_vals[0],max_vals[0],11), # mu
        np.linspace(min_vals[1],max_vals[1],14)) # log_msqrtR

    biggrid = np.stack(biggrid_expanded,-1).reshape(-1,2)
    
    biggrid_dataframe=pd.DataFrame(biggrid,columns=["mu","log_msqrtR"])

    #raise ValueError("debug")


    for surrogate_key in surrogates:
        bigsur_out = surrogates[surrogate_key].evaluate(biggrid_dataframe)

        # find peaks in sd
        sd_expanded = bigsur_out["sd"].reshape(biggrid_expanded[0].shape)
        
        sd_peaks = ( (sd_expanded[1:-1,1:-1] > sd_expanded[0:-2,1:-1]) &
                     (sd_expanded[1:-1,1:-1] > sd_expanded[2:,1:-1]) &
                     (sd_expanded[1:-1,1:-1] > sd_expanded[1:-1,0:-2]) &
                     (sd_expanded[1:-1,1:-1] > sd_expanded[1:-1,2:]))
        
        #sd_peaklocs = np.where(sd_peaks)
        
        sd_peakvals = sd_expanded[1:-1,1:-1][sd_peaks]
        sd_peak_mu = biggrid_expanded[0][1:-1,1:-1][sd_peaks]
        sd_peak_log_msqrtR = biggrid_expanded[1][1:-1,1:-1][sd_peaks]

        sd_peaksort = np.argsort(sd_peakvals)

        peakidx=-1
        for peakidx in range(min(2,sd_peakvals.shape[0])): # Use up to 2 peaks corresponding to relative maxima
            mu_val = sd_peak_mu[sd_peaksort][-peakidx-1]
            log_msqrtR_val = sd_peak_log_msqrtR[sd_peaksort][-peakidx-1]
            
            #raise ValueError("debug!")
            if only_on_gridlines_bool:
                (mu_val,
                 log_msqrtR_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                     mu_val,
                                                     log_msqrtR_val)
                
                pass

            plot_slices(_dest_href,
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
                        mu_val,log_msqrtR_val)
            pass
        
        peakidx += 1
    
        if peakidx < 2:
            # Did not display 2 peaks corresponding to relative maxima... Use peak corresponding to absolute maximum as well
            idx_absmax = np.argmax(sd_expanded)
            idxs_absmax = np.unravel_index(idx_absmax,sd_expanded.shape)

            mu_val = biggrid_expanded[0][idxs_absmax]
            log_msqrtR_val = biggrid_expanded[1][idxs_absmax]


            if only_on_gridlines_bool:
                (mu_val,
                 log_msqrtR_val) = snap_to_gridlines(surrogates[surrogate_key],
                                                     mu_val,
                                                     log_msqrtR_val)
                pass

            plot_slices(_dest_href,
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
                        mu_val,log_msqrtR_val)
        
            pass
        
        pass
    
    return {
        "dc:surrogate_eval": xmltreev(surrogate_eval_doc)
        }

