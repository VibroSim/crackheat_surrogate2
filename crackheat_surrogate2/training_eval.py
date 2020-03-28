import sys
import os
import numpy as np
import threading

from crackclosuresim2 import crackopening_from_tensile_closure
from crackclosuresim2 import load_closurestress
from crackclosuresim2 import crack_model_normal_by_name
from crackclosuresim2 import crack_model_shear_by_name


from angled_friction_model.angled_friction_model import angled_friction_model
from angled_friction_model.angled_friction_model import integrate_power



def afm_calc(params):
    try:
        (fixedparams,
         friction_coefficient_array,
         log_msqrtR_val) = params
    
        (bendingstress,dynamicnormalstress,dynamicshearstress,x_bnd_left,x_left,dx_left,x_bnd_right,x_right,dx_right,numdraws,E,nu,sigma_yield,tau_yield,closure_stress_left,crack_initial_opening_left,closure_stress_right,crack_initial_opening_right,tortuosity,aleft,aright,crack_model_normal_type,crack_model_shear_type,max_stddev,verbose,doplots) = fixedparams

        
        # Need to create the crack models here because functions can't be passed through multiprocessing.map()
        crack_model_normal=crack_model_normal_by_name(crack_model_normal_type,E,nu)
        crack_model_shear=crack_model_shear_by_name(crack_model_shear_type,E,nu)

        # This assumes crack_type=="quarterpenny" ... based on integrate_power() calculation
        slice_area_left = (x_left[1]-x_left[0]) * np.pi*abs(x_left)/2.0
        variance_weighting_left = slice_area_left**2.0
        slice_area_right = (x_right[1]-x_right[0]) * np.pi*abs(x_right)/2.0
        variance_weighting_right = slice_area_right**2.0
        
        if max_stddev is None:
            max_stddev_eachside = None
            pass
        else:
            max_stddev_eachside = max_stddev/np.sqrt(2.0)
            pass
    
        (power_per_m2_left,
         power_per_m2_stddev_left,
         vibration_ampl_left,
         shear_vibration_ampl_left,
         residual_sub_left,
         residual_add_left) = angled_friction_model(x_bnd_left,x_left,dx_left,
                                                            numdraws,
                                                            E,nu,
                                                            sigma_yield,tau_yield,
                                                            friction_coefficient_array,
                                                            closure_stress_left,
                                                            crack_initial_opening_left,
                                                            tortuosity*np.pi/180.0,
                                                            aleft,
                                                            bendingstress,
                                                            dynamicnormalstress,
                                                            dynamicshearstress, 
                                                            1.0, # vibration_frequency.... we use 1.0 for frequency so as to obtain heating per cycle
                                                            crack_model_normal,
                                                            crack_model_shear,
                                                            1.0, # crack_model_shear_factor currently hardwired to 1.0
                                                            np.exp(log_msqrtR_val),
                                                            "quarterpenny",
                                                            None,
                                                            verbose,
                                                            doplots,
                                                            max_total_stddev = max_stddev_eachside)

        (power_per_m2_right,
         power_per_m2_stddev_right,
         vibration_ampl_right,
         shear_vibration_ampl_right,
         residual_sub_right,
         residual_add_right) = angled_friction_model(x_bnd_right,x_right,dx_right,
                                                             numdraws,
                                                             E,nu,
                                                             sigma_yield,tau_yield,
                                                             friction_coefficient_array,
                                                             closure_stress_right,
                                                             crack_initial_opening_right,
                                                             tortuosity*np.pi/180.0,
                                                             aright,
                                                             bendingstress,
                                                             dynamicnormalstress,
                                                             dynamicshearstress, # vib_shear_stress_ampl, 
                                                             1.0, # vibration_frequency.... we use 1.0 for frequency so as to obtain heating per cycle
                                                             crack_model_normal,
                                                             crack_model_shear,
                                                             1.0,
                                                             np.exp(log_msqrtR_val),
                                                             "quarterpenny",
                                                             None,
                                                             verbose,
                                                             doplots,
                                                             max_total_stddev = max_stddev_eachside)
        
        #(totalpower[testgridpos:(testgridpos+count)],
        # totalpower_stddev[testgridpos:(testgridpos+count)]) = integrate_power(xrange,power_per_m2_left,power_per_m2_left_stddev) + integrate_power(xrange,power_per_m2_right,power_per_m2_right_stddev)

        (totalpower_left,
         totalpower_stddev_left) = integrate_power(x_left,"quarterpenny",None,power_per_m2_left,power_per_m2_stddev_left)
        (totalpower_right,
         totalpower_stddev_right) = integrate_power(x_right,"quarterpenny",None,power_per_m2_right,power_per_m2_stddev_right)
        
        totalpower=totalpower_left + totalpower_right
        totalpower_stddev = np.sqrt(totalpower_stddev_left**2.0 + totalpower_stddev_right**2.0)

        soft_closure_diags = (residual_sub_left,residual_add_left,residual_sub_right,residual_add_right) # soft closure diagnostics
        pass
        

    except:
        import traceback
        traceback.print_exc()
        raise
    

    return (totalpower,totalpower_stddev,soft_closure_diags)

# To run interactively from R, paste in: 
rinteractive = r"""
py$testgrid = testgrid
py$tortuosity = tortuosity
py$leftclosure = leftclosure
py$rightclosure = rightclosure
py$aleft = aleft
py$aright = aright
py$sigma_yield = sigma_yield
py$tau_yield = tau_yield
py$crack_model_normal_type = crack_model_normal_type
py$crack_model_shear_type = crack_model_shear_type
py$E = E
py$nu = nu
py$numdraws = numdraws
py$multiprocessing_pool = NULL
repl_python()
from crackheat_surrogate.training_eval import training_eval
training_eval_output = training_eval(testgrid,tortuosity,leftclosure,rightclosure,aleft,aright,sigma_yield,tau_yield,crack_model_normal_type,crack_model_shear_type,E,nu,numdraws)

"""

def training_eval(testgrid,bendingstress,dynamicnormalstress,dynamicshearstress,tortuosity,leftclosure,rightclosure,aleft,aright,sigma_yield,tau_yield,crack_model_normal_type,crack_model_shear_type,E,nu,numdraws,max_stddev = None,multiprocessing_pool = None,multiprocessing_lock = None):
    """ NOTE: tortuosity should be angular standard deviation in degrees!!!"""
    #print("bendingstress=%s" % (str(bendingstress)))
    log_mu=np.array(testgrid["log_mu"],dtype='d')
    log_msqrtR=np.array(testgrid["log_msqrtR"],dtype='d')

    if multiprocessing_lock is None:
        multiprocessing_lock = threading.Lock() # Won't do anything since it is private
        pass

    verbose=False
    doplots=False
    

    # TODO: Should probably put assumed crack opening displacement in the input csv, rather than recalculating it here
    # TODO: should probably use Pandas to read .CSV file (see soft_closure_paper)
    #leftclosure_data = np.loadtxt(leftclosure,skiprows=1,delimiter=',')
    #
    ## expand out position base to one sample beyond crack end
    #assert(leftclosure_data.shape[1]==2)
    #dx_left = leftclosure_data[1,0]-leftclosure_data[0,0]
    #x_left = np.concatenate((leftclosure_data[:,0],np.array((leftclosure_data[-1,0]+dx_left,))))
    #x_bnd_left = np.concatenate((x_left-dx_left/2.0,np.array((x_left[-1]+dx_left/2.0,x_left[-1]+3*dx_left/2.0))))
    #if x_bnd_left[0] < 0.0:
    #    x_bnd_left[0]=0.0
    #    pass
    #closure_stress_left = np.concatenate((leftclosure_data[:,1],np.array((0.0,))))

    (x_left,
     x_bnd_left,
     dx_left,
     aleft_verify,
     closure_stress_left,
     crack_opening_left_notpresent) = load_closurestress(leftclosure)

    if aleft_verify is not None: # crack lengths should match
        assert((aleft_verify-aleft)/aleft < 1e-2)
        pass

    #rightclosure_data = np.loadtxt(rightclosure,skiprows=1,delimiter=',')
    #
    ## expand out position base to one sample beyond crack end
    #assert(rightclosure_data.shape[1]==2)
    #dx_right = rightclosure_data[1,0]-rightclosure_data[0,0]
    #x_right = np.concatenate((rightclosure_data[:,0],np.array((rightclosure_data[-1,0]+dx_right,))))
    #x_bnd_right = np.concatenate((x_right-dx_right/2.0,np.array((x_right[-1]+dx_right/2.0,x_right[-1]+3*dx_right/2.0))))
    #if x_bnd_right[0] < 0.0:
    #    x_bnd_right[0]=0.0
    #    pass
    #closure_stress_right = np.concatenate((rightclosure_data[:,1],np.array((0.0,))))

    (x_right,
     x_bnd_right,
     dx_right,
     aright_verify,
     closure_stress_right,
     crack_opening_right_notpresent) = load_closurestress(rightclosure)

    if aright_verify is not None: # crack lengths should match
        assert((aright_verify-aright)/aright < 1e-2)
        pass
        

    crack_model_normal=crack_model_normal_by_name(crack_model_normal_type,E,nu)
    #crack_model_shear=crack_model_shear_by_name(crack_model_shear_type,E,nu)

    # Evaluate initial crack opening gaps from extrapolated tensile closure field
    crack_initial_opening_left = crackopening_from_tensile_closure(x_left,x_bnd_left,closure_stress_left,dx_left,aleft,sigma_yield,crack_model_normal)

    crack_initial_opening_right = crackopening_from_tensile_closure(x_right,x_bnd_right,closure_stress_right,dx_right,aright,sigma_yield,crack_model_normal)


    totalpower = np.zeros(log_mu.shape[0],dtype='d')
    totalpower_stddev = np.zeros(log_mu.shape[0],dtype='d')
    
    # Find unique values of bendingstress, dynamicstress, and msqrtRn
    #print("bendingstress=%s" % (str(bendingstress)))
    #print("dynamicstress=%s" % (str(dynamicstress)))
    #print("msqrtR=%s" % (str(msqrtR)))

    fixedparams = (bendingstress,dynamicnormalstress,dynamicshearstress,x_bnd_left,x_left,dx_left,x_bnd_right,x_right,dx_right,numdraws,E,nu,sigma_yield,tau_yield,closure_stress_left,crack_initial_opening_left,closure_stress_right,crack_initial_opening_right,tortuosity,aleft,aright,crack_model_normal_type,crack_model_shear_type,max_stddev,verbose,doplots) 

    (uniquevals,uniquecounts) = np.unique(np.array((log_msqrtR,),dtype='d'),axis=1,return_counts=True)

    #print("uniquevals=%s" % (str(uniquevals)))
    # build dictionary of counts
    countdict = {}
    for uniqueidx in range(uniquevals.shape[1]):
        key=(uniquevals[0,uniqueidx],)
        countdict[key] = uniquecounts[uniqueidx]
        pass
    

    testgridpos = 0
    paramlist = []
    
    #print("countdict keys=%s" % (str(list(countdict.keys()))))
    while testgridpos < log_msqrtR.shape[0]:
        key = (log_msqrtR[testgridpos],)
        count = countdict[key]

        # If one of these asserts fails, it probably means the test grid has duplicated values of (bendingstress,dynamicstress,log_msqrtR) that are not contiguous
        #assert((bendingstress[testgridpos:(testgridpos+count)]==bendingstress[testgridpos]).all())
        #assert((dynamicstress[testgridpos:(testgridpos+count)]==dynamicstress[testgridpos]).all())
        assert((log_msqrtR[testgridpos:(testgridpos+count)]==log_msqrtR[testgridpos]).all())

        friction_coefficient=np.exp(log_mu[testgridpos:(testgridpos+count)])

        params = (fixedparams,friction_coefficient,log_msqrtR[testgridpos])

        paramlist.append(params)
        testgridpos += count
        pass


    #
    if multiprocessing_pool is None:
        resultlist = list(map(afm_calc,paramlist))
        pass
    else:
        #  Because multiprocessing.Pool() is rumored to be thread-unsafe
        # we decompose it into asynchronous operations that are protected by multiprocessing_lock
        #resultlist = multiprocessing_pool.map(afm_calc,paramlist)
        multiprocessing_lock.acquire()
        asyncresult = multiprocessing_pool.map_async(afm_calc,paramlist)
        multiprocessing_lock.release()

        # asyncresult.wait() ought to be safe because it is a call to the threading library's wait() function under the hood
        asyncresult.wait()
        multiprocessing_lock.acquire()
        try:
            resultlist = list(asyncresult.get())
            pass
        finally:
            multiprocessing_lock.release()
            pass
        pass
    
    totalpower_toconcat=[]
    totalpower_stddev_toconcat=[]
    soft_closure_diags_toarray=[]
    
    
    for (totalpower,totalpower_stddev,soft_closure_diags) in resultlist:
        totalpower_toconcat.append(totalpower)
        totalpower_stddev_toconcat.append(totalpower_stddev)
        soft_closure_diags_toarray.append(soft_closure_diags)
        pass

    

    return (np.concatenate(totalpower_toconcat),np.concatenate(totalpower_stddev_toconcat),np.array(soft_closure_diags_toarray,dtype='d'))
