import sys
import os
import copy
import threading

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats

import json

# NOTE:
# Assumes:
#  * No nugget,
#  * Universal kriging
#  * only pre-programmed formulas
#  * only Matern5_2 covariance
#  * no bias correction

accel_trisolve_opencl_code=r"""

__kernel void perform_multiply(__global float *Ttranspose_inverse,__global float *new_positions_covariance,__global float *product,unsigned ncols)
{
  unsigned rownum = get_global_id(0);
  unsigned colnum = get_global_id(1);

  product[rownum*ncols + colnum] = Ttranspose_inverse[rownum*ncols + colnum]*new_positions_covariance[colnum];
}

__kernel void perform_add(__global float *product,__global float *solution,unsigned ncols)
{
  unsigned rownum = get_global_id(0);
  unsigned rowoffs;
  unsigned colnum;
  float accum=0.0;

  rowoffs = rownum*ncols;

  for (colnum=0;colnum < ncols;colnum++) {  
    accum += product[rowoffs + colnum];
  }
  solution[rownum]=accum;
}

"""


def covMatern5_2(x1,x2,param,scaling_factor,var):

    ecart = np.abs(x1[:,np.newaxis,:] - x2[np.newaxis,:,:]) / (param[np.newaxis,np.newaxis,:] / scaling_factor)
    toadd = ecart - np.log(1.0+ecart+ecart**2.0/3.0)
    
    s=np.sum(toadd,axis=2)
    
    return(np.exp(-s) * var);

def covMatern5_2_deriv_x2_k(x1,x2,params,scaling_factor,var,k):
    # x1 and x2 are arrays for which the second (index 1) axis
    # lengths match. k is an index into that axis

    k_index = {"log_mu_norm": 0, "log_msqrtR_norm": 1}[k]

    # matern 5/2 covariance is a product
    # of factors corresponding to the different indices into the
    # second axis of x1 and x2

    # The covariance is expressed as:
    #   cov = exp(-ecart[k])*exp(log(1+ecart[k]+(ecart[k]**2)/3)) * ...
    # or equivalently
    #   cov = exp(-ecart[k]) * (1 + ecart[k] + (ecart[k]**2)/3) * ...
    # where
    #   ecart[k] = |x1[k]-x2[k]|/(param[k]/sqrt(5))
    # where sqrt(5) is the scaling_factor.
    #
    # There is also a pre-multiplier "var" representing the variance
    #
    # Evaluation of the derivative :
    #   dcov/decart[k] = -exp(-ecart[k])* (1 + ecart[k] + (ecart[k]**2)/3) * ...
    #    + exp(-ecart[k]) * (1 + 2*ecart[k]/3) * ... (... represents factors for indices other than k, and also the var factor)
    #
    #  In each of the terms we can substitute cov back in: 
    #    dcov/decart[k] = -cov + cov*(1 + 2*ecart[k]/3)/(1 + ecart[k] + (ecart[k]**2)/3)
    #  Find a common denominator for the two terms:
    #    dcov/decart[k] = -cov*(1 + ecart[k] + (ecart[k]**2)/3)/(1 + ecart[k] + (ecart[k]**2)/3) + cov*(1 + 2*ecart[k]/3)/(1 + ecart[k] + (ecart[k]**2)/3)
    #  Now add the terms
    #    dcov/decart[k] = - cov*(1*ecart[k]/3 + (ecart[k]**2)/3)/(1 + ecart[k] + (ecart[k]**2)/3)
    #  To check, the derivative should be zero at ecart[k] = 0
    #   ... which it is!
    #  Also need
    #    decart[k]/dx2[k] = -sign(x1[k]-x2[k])/(param[k]/scalefactor)
    #
    #  Chain rule to put these together
    #    dcov/dx2[k] = dcov/decart[k] * decart[k]/dx2[k]
    #    dcov/dx2[k] =  cov*(1*ecart[k]/3 + (ecart[k]**2)/3)/(1 + ecart[k] + (ecart[k]**2)/3) * sign(x1[k]-x2[k])/(param[k]/scalefactor)

    ecart_k = np.abs(x1[:,np.newaxis,k_index] - x2[np.newaxis,:,k_index]) / (params[k_index] / scaling_factor)
    sign_k = np.sign(x1[:,np.newaxis,k_index] - x2[np.newaxis,:,k_index])
    
    return (covMatern5_2(x1,x2,params,scaling_factor,var)*(ecart_k/3.0 + (ecart_k**2.0)/3.0)/(1.0 + ecart_k + (ecart_k**2.0)/3.0))*sign_k/(params[k_index]/scaling_factor)


# centralize contexts/queues by thread so we don't create so many and overflow the low fixed limit of NVIDIA devices
clcontext_queues_bythread={} # dictionary indexed by threading.current_thread().ident of dictionary indexed by device name of (context,queue)


class surrogate_model(object):
    # Class representing raw DiceKriging surrogate model
    closure_lowest_avg_load_used = None # Lowest load, in Pa, with data that went into closure model; in some cases may be a bound on the validity of the model/training data

    # DiceKriging outputs
    X = None
    y = None
    T = None
    Ttranspose_inverse = None # cached for accel_trisolve option
    #clcontext_queues=None # dictionary by tuple of device pointer ints of (context,queue)
    clbuffers=None # dictionary by id(context) of (Ttranspose_inverse,new_positions_covariance_buf,product_buf,solution_buf)
    #clprg = None # dictionary by id(context) of opencl program
    clkern = None # dictionary by id(context) of (multply opencl kernel,add opencl kernel)
    
    z = None
    M = None
    beta = None

    trend_formula = None
    
    covariance_class = None
    covariance_name = None
    covariance_paramset_n = None
    
    covariance_sd2 = None
    covariance_param = None

    # Precalculated T_M
    T_M = None
    
    def __init__(self,**kwargs):
        for argname in kwargs:
            if hasattr(self,argname):
                setattr(self,argname,kwargs[argname])
                pass
            else:
                raise ValueError("Unknown attribute: %s" % (argname))
            pass

        # Precalculate T_M
        self.T_M = scipy.linalg.cholesky(np.dot(self.M.T,self.M),lower=False)

        pass

    def eval_formula_values(self,new_positions):
        # Can only handle pre-programmed formulas...
        if self.trend_formula==[u'~x + I(x^2)']: # Simple linear + quadratic
            formula_values=np.array((np.ones(new_positions.shape[0],dtype='d'),
                                     new_positions["x"],
                                     new_positions["x"]**2.0),dtype='d').T
            pass
        elif self.trend_formula==['~log_mu_norm + log_msqrtR_norm + I(log_mu_norm^2) + I(log_msqrtR_norm^2) + ',
                                  '    I(log_mu_norm * log_msqrtR_norm)']:
             # Full linear and quadratic in log_mu and log_msqrtR, all normalized 
            formula_values=np.array((np.ones(new_positions.shape[0],dtype='d'),
                                     new_positions["log_mu_norm"],
                                     new_positions["log_msqrtR_norm"],
                                     new_positions["log_mu_norm"]**2.0,
                                     new_positions["log_msqrtR_norm"]**2.0,
                                     new_positions["log_mu_norm"]*new_positions["log_msqrtR_norm"]),dtype='d').T
            pass
        else:
            raise ValueError("Unknown formula: %s" % ("".join(self.trend_formula)))
        return formula_values

    def eval_formula_values_deriv_k(self,new_positions,k):
        # Can only handle pre-programmed formulas...
        if self.trend_formula==['~log_mu_norm + log_msqrtR_norm + I(log_mu_norm^2) + I(log_msqrtR_norm^2) ',
                                  '    + I(log_mu_norm * log_msqrtR_norm)']:
            # Full linear and quadratic in log_mu and log_msqrtR, all normalized
            if k=="log_mu_norm":
                formula_values_deriv_k=np.array((np.zeros(new_positions.shape[0],dtype='d'),
                                                 np.ones(new_positions.shape[0],dtype='d'),
                                                 np.zeros(new_positions.shape[0],dtype='d'),
                                                 2.0*new_positions["log_mu_norm"],
                                                 np.zeros(new_positions.shape[0],dtype='d'),
                                                 new_positions["log_msqrtR_norm"]),dtype='d').T
                pass
            elif k=="log_msqrtR_norm":
                formula_values_deriv_k=np.array((np.zeros(new_positions.shape[0],dtype='d'),
                                                 np.zeros(new_positions.shape[0],dtype='d'),
                                                 np.ones(new_positions.shape[0],dtype='d'),
                                                 np.zeros(new_positions.shape[0],dtype='d'),
                                                 2.0*new_positions["log_msqrtR_norm"],
                                                 new_positions["log_mu_norm"]),dtype='d').T
                pass
            else:
                raise ValueError("k can only be (normalized) log_mu, log_msqrtR. ")
            pass
        else:
            raise ValueError("Unknown formula: %s" % ("".join(self.trend_formula)))
        return formula_values_deriv_k


    def solve_triangular_Ttranspose(self,rhs,accel_trisolve_devs=None):
        """ solve self.T.T*result = rhs, where self.T.T is lower triangular,
        using GPU if requested"""

        if accel_trisolve_devs is None or accel_trisolve_devs[0] == os.getpid():
            # Disable GPU as it hasn't been specified or
            # we don't want to use it in the main process
            # lest we have problems after a fork()
            result = scipy.linalg.solve_triangular(self.T.T,rhs,lower=True)
            return result
        
        
        import pyopencl as cl
        mf=cl.mem_flags

        #if self.clcontext_queues is None:
        #    #print("opencl initializing pid %d\n" % (os.getpid()))
        #    self.clcontext_queues={}
        #    pass
        
        if threading.current_thread().ident not in clcontext_queues_bythread:
            clcontext_queues_bythread[threading.current_thread().ident] = {}
            pass
            
        if accel_trisolve_devs[1] not in clcontext_queues_bythread[threading.current_thread().ident]:
            platforms=cl.get_platforms()
            platforms_byname = { platform.name: platform for platform in platforms }
            devicelist = []
            for dev in accel_trisolve_devs[1]:
                (platformname,devicename) = dev
                platform = platforms_byname[platformname]
                devices=platform.get_devices()
                devices_byname = { device.name: device for device in devices }
                device=devices_byname[devicename]
                devicelist.append(device)
                pass

            context = cl.Context(devices=devicelist)
            queue = cl.CommandQueue(context)
            clcontext_queues_bythread[threading.current_thread().ident][accel_trisolve_devs[1]] = (context,queue)
            pass
            
        (context,queue) = clcontext_queues_bythread[threading.current_thread().ident][accel_trisolve_devs[1]]
            
        if self.Ttranspose_inverse is None:
            self.Ttranspose_inverse = np.ascontiguousarray(np.linalg.inv(self.T.T).astype(np.float32))
            self.clbuffers={}
            self.clkern={}
            pass

        if not id(context) in self.clbuffers:
            Ttranspose_inverse_buf = cl.Buffer(context,mf.READ_ONLY|mf.COPY_HOST_PTR,hostbuf=self.Ttranspose_inverse)
            print("Ttranspose_inverse_bufsize=%d" % (Ttranspose_inverse_buf.size))
            rhs_buf=cl.Buffer(context,mf.READ_ONLY,size=self.Ttranspose_inverse.shape[1]*np.float32().itemsize)
            print("rhs_bufsize=%d" % (rhs_buf.size))
            product_buf=cl.Buffer(context,mf.READ_WRITE,size=self.Ttranspose_inverse.shape[0]*self.Ttranspose_inverse.shape[1]*np.float32().itemsize)
            print("product_bufsize=%d" % (product_buf.size))
            solution_buf = cl.Buffer(context,mf.WRITE_ONLY,size=self.Ttranspose_inverse.shape[0]*np.float32().itemsize)
            print("solution_bufsize=%d" % (solution_buf.size))

            self.clbuffers[id(context)] = (Ttranspose_inverse_buf,rhs_buf,product_buf,solution_buf)
            pass

        
        (Ttranspose_inverse_buf,rhs_buf,product_buf,solution_buf) = self.clbuffers[id(context)]

        if not id(context) in self.clkern:
            
            clprg = cl.Program(context,accel_trisolve_opencl_code)
            clprg.build()
            
            perform_multiply_kern = clprg.perform_multiply
            perform_add_kern = clprg.perform_add

            perform_multiply_kern.set_scalar_arg_dtypes([None,None,None,np.uint32])
            perform_add_kern.set_scalar_arg_dtypes([None,None,np.uint32])
            
            self.clkern[id(context)]=(perform_multiply_kern,perform_add_kern)
            pass
            
        (perform_multiply_kern,perform_add_kern) = self.clkern[id(context)]
            


        # Transfer the parameter, rhs, (was new_positions_covariance), to the GPU
        rhs_array = np.ascontiguousarray(rhs.astype(np.float32))
        copy_ev = cl.enqueue_copy(queue,rhs_buf,rhs_array)
        
        
        
        # Peform multiply
        multiply_ev = perform_multiply_kern(queue,self.Ttranspose_inverse.shape,None,Ttranspose_inverse_buf,rhs_buf,product_buf,self.Ttranspose_inverse.shape[1],wait_for=(copy_ev,))
        
        queue.flush()
            
        # perform add
        add_ev = perform_add_kern(queue,(self.Ttranspose_inverse.shape[0],),None,product_buf,solution_buf,self.Ttranspose_inverse.shape[1],wait_for=(multiply_ev,))
        
        # copy result back
        result=np.empty(self.Ttranspose_inverse.shape[0],dtype=np.float32)
        copy_res_ev = cl.enqueue_copy(queue,result,solution_buf,wait_for=(add_ev,))
        queue.finish()
            
        return result
    
    def evaluate(self,new_positions,meanonly=False,accel_trisolve_devs=None):
        """accel_trisolve_devs is either None (disabling GPU acceleration)
or a tuple (filterpid,device_list) where GPU acceleration will only occur
when the return value of os.getpid() does NOT match filterpid. device_list
should be a tuple of (platform name, device name) tuples.

If you have a pyopencl context, you can create the necessary parameter
from:
 accel_trisolve_devs=(-1,tuple([ (dev.platform.name,dev.name) for dev in context.devices]))

The reason for the filterpid parameter is that if you are calling evaluate() 
in parallel using Python multiprocessing, OpenCL will only work properly 
if imported in the subprocesses but NOT in the parent. So a workaround
is to pre-evaluate accel_trisolve_devs, and assign that value in the parent
while replacing the -1 with the parent process id. That way if the 
parent calls evaluate() it will just use the slow triangular evaluation
and OpenCL will not be imported, but the subprocesses can all use the
fast path. 
"""

        formula_values = self.eval_formula_values(new_positions)
        predicted_trend = np.einsum("ij,j",formula_values,self.beta)

        assert(self.covariance_class==u'covTensorProduct') # Can only handle covTensorProduct implementation of the covariance

        assert(self.covariance_name==[u"matern5_2"]) # can only handle matern5_2 covariance

        assert(self.covariance_paramset_n==[1]) # only handle covariance depending on ranges, not exponent (powexp)


        # covariance of new positions and training data positions
        new_positions_covariance = covMatern5_2(self.X,new_positions[["log_mu_norm","log_msqrtR_norm"]].values,self.covariance_param,np.sqrt(5.0),self.covariance_sd2)  # The sqrt(5.0) is the scaling factor for Matern5_2

        # Solve self.T.T * new_positions_solution = new_positions_covariance
        new_positions_solution = self.solve_triangular_Ttranspose(new_positions_covariance,accel_trisolve_devs=accel_trisolve_devs)
        
        yhat = predicted_trend + np.inner(new_positions_solution.T,self.z)

        if not meanonly: 

            s2_predict_1 = np.einsum('ij,ij->j',new_positions_solution,new_positions_solution)

        

            s2_predict_mat = scipy.linalg.solve_triangular(self.T_M.T, (formula_values - np.dot(new_positions_solution.T,self.M)).T, lower = True)

            s2_predict_2 = np.einsum('ij,ij->j',s2_predict_mat,s2_predict_mat)
            
            s2_predict = np.maximum(self.covariance_sd2 - s2_predict_1 + s2_predict_2, 0.0)
            
            q95 = scipy.stats.norm.ppf(.975) 
            
            lower95 = yhat - q95*np.sqrt(s2_predict)
            upper95 = yhat + q95*np.sqrt(s2_predict)
            
        
            resdict = {
                "trend": predicted_trend,
                "mean": yhat,
                "c": new_positions_covariance,
                "Tinv.c": new_positions_solution,
                "sd": np.sqrt(s2_predict),
                "lower95": lower95,
                "upper95": upper95,
            }
            pass
        else:
            resdict = {
                "mean": yhat   
            }
            pass
        return resdict


    def evaluate_derivative(self,new_positions,k,accel_trisolve_devs=None):
        # new_positions is a dataframe.... indexed i,k
        # Evaluate derivative with respect to column "k" of new positions
        
        formula_values_deriv_k = self.eval_formula_values_deriv_k(new_positions,k) # indexed i,j

        # trend[i] = dot(formula_values, beta) = sum_j formula_values[i,j]*beta[j]
        # dtrend[i]/dnew_positions_k = sum_j dtrend[i]_dformula_values[i,j]  * dformula_values[i,j]_dnew_positions_k
        # dtrend[i]_dformula_values[i,j] = beta[j]
        # dformula_values[i,j]_dnew_positions_k = formula_values_deriv_k[i,j]

        # dtrend[i]/dnew_positions_k = sum_j beta[j]  * formula_values_deriv_k[i,j]
        
        predicted_trend_deriv_k = np.einsum("ij,j",formula_values_deriv_k,self.beta)  # dpredicted_trend_dnew_positions_k

        assert(self.covariance_class==u'covTensorProduct') # Can only handle covTensorProduct implementation of the covariance

        assert(self.covariance_name==[u"matern5_2"]) # can only handle matern5_2 covariance

        assert(self.covariance_paramset_n==[1]) # only handle covariance depending on ranges, not exponent (powexp)


        # covariance of new positions and training data positions
        #new_positions_covariance = covMatern5_2(self.X,new_positions,self.covariance_param,np.sqrt(5.0),self.covariance_sd2)  # The sqrt(5.0) is the scaling factor for Matern5_2

        new_positions_covariance_deriv_k = covMatern5_2_deriv_x2_k(self.X,new_positions[["log_mu_norm","log_msqrtR_norm"]].values,self.covariance_param,np.sqrt(5.0),self.covariance_sd2,k)  # The sqrt(5.0) is the scaling factor for Matern5_2
        # row index of new_positions_covariance corresponds to rows of self.X; column index of new_positions_covariance correspond to rows of new_positions

        # T' * new_positions_solution = new_positions_covariance  -> want derivative of new_positions_solution with respect to new_positions_covariance (really with respect to new_positions[:,k]
        #  inv(T')*(T') * new_positions_solution = inv(T')*new_positions_covariance
        # new_positions_solution = inv(T')*new_positions_covariance
        # new_positions_solution[i] = sum_j inv_TT[i,j]*new_positions_covariance[j]
        # Since new_positions_covariance is really a matrix (columns represent different positions we are calc'ing in parallel)
        # new_positions_solution[i,l] = sum_j inv_TT[i,j]*new_positions_covariance[j,l]

        # derivative of new_positions_solution[i,l] with respect to new_positions_covariance[j,l] = inv_TT[i,j]

        # yhat[l] = predicted_trend + np.inner(new_positions_solution.T,self.z) = predicted_trend[l] + sum_i new_positions_solution[i,l]*self.z[i]
        # dyhat_dnew_positions_k = dpredicted_trend_dnew_positions_k[l] + sum_i self.z[i]*dnew_positions_solution[i,l]/dnew_positions_k[l]
        # dnew_positions_solution[i,l]/dnew_positions_k[l] = sum_j dnew_positions_solution[i,l]/dnew_positions_covariance[j,l] * dnew_positions_covariance[j,l]/dnew_positions_k[l]
        # dnew_positions_solution[i,l]/dnew_positions_k[l] = sum_j inv_TT[i,j] * dnew_positions_covariance[j,l]/dnew_positions_k[l]
        # dnew_positions_solution[i,l]/dnew_positions_k[l] = sum_j inv_TT[i,j] * new_positions_covariance_deriv_k[j,l]

        # Since we can calculate new_positions_solution[i,l] (formula above) with solve_triangular: 
        #new_positions_solution = scipy.linalg.solve_triangular(self.T.T,new_positions_covariance,lower=True) # !!!!*** solve_triangular should be delegated to GPU
        # .... We can do the same with new_positions_solution_deriv_k, since the structure is the same
        #new_positions_solution_deriv_k = scipy.linalg.solve_triangular(self.T.T,new_positions_covariance_deriv_k,lower=True) # !!!!*** solve_triangular should be delegated to GPU
        # Solve self.T.T * new_positions_solution_deriv_k = new_positions_covariance_deriv_k
        new_positions_solution_deriv_k = self.solve_triangular_Ttranspose(new_positions_covariance_deriv_k,accel_trisolve_devs=accel_trisolve_devs)

        
        
        #yhat = predicted_trend + np.inner(new_positions_solution.T,self.z)
        yhat_deriv_k = predicted_trend_deriv_k + np.inner(new_positions_solution_deriv_k.T,self.z)
        
        return yhat_deriv_k



    
    @classmethod
    def fromjson(cls,model_json,**kwargs):

        # Extract X,y, T, z, M, and beta
        X = np.array(model_json["attributes"]["X"]["value"],dtype='d').reshape(model_json["attributes"]["X"]["attributes"]["dim"]["value"],order='F')
        
        y = np.array(model_json["attributes"]["y"]["value"],dtype='d').reshape(model_json["attributes"]["y"]["attributes"]["dim"]["value"],order='F')

        T = np.array(model_json["attributes"]["T"]["value"],dtype='d').reshape(model_json["attributes"]["T"]["attributes"]["dim"]["value"],order='F')

        z = np.array(model_json["attributes"]["z"]["value"],dtype='d')

        M = np.array(model_json["attributes"]["M"]["value"],dtype='d').reshape(model_json["attributes"]["M"]["attributes"]["dim"]["value"],order='F')

        beta = np.array(model_json["attributes"]["trend.coef"]["value"],dtype='d')

        trend_formula = model_json["attributes"]["trend.formula"]["value"]

        covariance_class = model_json["attributes"]["covariance"]["value"]["class"]
        covariance_name = model_json["attributes"]["covariance"]["attributes"]["name"]["value"]
        covariance_paramset_n = model_json["attributes"]["covariance"]["attributes"]["paramset.n"]["value"]

        
        covariance_range = model_json["attributes"]["covariance"]["attributes"]["range.val"]["value"]

        covariance_sd2 = model_json["attributes"]["covariance"]["attributes"]["sd2"]["value"][0]
        
        covariance_param=np.array(covariance_range,dtype='d')



        
        return cls(X=X,
                   y=y,
                   T=T,
                   z=z,
                   M=M,
                   beta=beta,
                   trend_formula=trend_formula,
                   covariance_class=covariance_class,
                   covariance_name=covariance_name,
                   covariance_paramset_n=covariance_paramset_n,
                   covariance_sd2=covariance_sd2,
                   covariance_param=covariance_param,
                   **kwargs)

    @classmethod
    def fromjsonfile(cls,filename,**kwargs):
        fh = open(filename)
        jsondata = json.load(fh)
        fh.close()
        return cls.fromjson(jsondata,**kwargs)
        
    pass


class denormalized_surrogate(surrogate_model):
    # Class representing denormalized surrogate model
    # that operates on physical units

    params_nominal = None
    output_nominal = None
    

    bendingstress = None
    dynamicnormalstressampl = None
    dynamicshearstressampl = None
    thermalpower = None
    excfreq = None

    def evaluate(self,new_positions,meanonly=False,accel_trisolve_devs=None):

        normalized_new_positions = pd.DataFrame({"log_mu_norm": new_positions["log_mu"]/self.params_nominal["log_mu"],
                                                 "log_msqrtR_norm": new_positions["log_msqrtR"]/self.params_nominal["log_msqrtR"] })
        
        resdict = super(denormalized_surrogate,self).evaluate(normalized_new_positions,meanonly=meanonly,accel_trisolve_devs=accel_trisolve_devs)

        resdict["mean"] *= self.output_nominal
        if not meanonly: 
            resdict["trend"] *= self.output_nominal
            resdict["c"] *= self.output_nominal
            resdict["Tinv.c"] *= self.output_nominal
            resdict["sd"] *= self.output_nominal
            resdict["lower95"] *= self.output_nominal
            resdict["upper95"] *= self.output_nominal
            pass
        
        return resdict


    def evaluate_derivative(self,new_positions,k,accel_trisolve_devs=None):

        k_norm = {"log_mu": "mu_norm", "log_msqrtR": "log_msqrtR_norm" }[k]

        normalized_new_positions = pd.DataFrame({"log_mu_norm": new_positions["log_mu"]/self.params_nominal["log_mu"],
                                                 "log_msqrtR_norm": new_positions["log_msqrtR"]/self.params_nominal["log_msqrtR"] })
        
        derivative_wrt_normalized = super(denormalized_surrogate,self).evaluate_derivative(normalized_new_positions,k_norm,accel_trisolve_devs=accel_trisolve_devs)
        
        # So the superclass evaluates
        #   surrogate(coordinates/coordinates_nominal)/output_nominal
        # and in evaluate() we multiply by output_nominal
        # to get the correct output
        #
        #
        # Here we want the derivative of
        # [surrogate(coordinates/coordinates_nominal)/output_nominal] * output_nominal
        # with respect to element k of coordinates
        # so we want
        # d[surrogate(coordinates/coordinates_nominal)/output_nominal]/dcoordinate_k * output_nominal
        # where coordinate_k/coordinate_nominal_k = coordinate_normalized_k and
        # d[surrogate(coordinates/coordinates_nominal)/output_nominal]/dcoordinate_k
        # =d[surrogate(coordinates/coordinates_nominal)/output_nominal]/d(coordinate_normalized_k*coordinate_nominal_k)
        # =d[surrogate(coordinates/coordinates_nominal)/output_nominal]/d(coordinate_normalized_k)  / coordinate_nominal_k)
        # where d[surrogate(coordinates/coordinates_nominal)/output_nominal]/d(coordinate_normalized_k) = 
        #  super(denormalized_surrogate,self).evaluate_derivative(normalized_new_positions,k)

        # so result is super(denormalized_surrogate,self).evaluate_derivative(normalized_new_positions,k) * output_nominal / coordinate_nominal_k
        return derivative_wrt_normalized * self.output_nominal / self.params_nominal[k]
        


    
        
    @classmethod
    def fromjson(cls,surrogate_json,closure_lowest_avg_load_used_default=np.nan,**kwargs):
        
        #surrogate_json = json.load(open('/tmp/test.json'))
        params_nominal_index = surrogate_json["attributes"]["names"]["value"].index('params_nominal')
        #params_nominal = np.array(surrogate_json["value"][params_nominal_index]["value"],dtype='d')
        params_nominal = {
            "log_mu": surrogate_json["value"][params_nominal_index]["value"][0],
            "log_msqrtR": surrogate_json["value"][params_nominal_index]["value"][1],
        }
        output_nominal_index = surrogate_json["attributes"]["names"]["value"].index('output_nominal')
        output_nominal = np.array(surrogate_json["value"][output_nominal_index]["value"],dtype='d')
        model_json_index = surrogate_json["attributes"]["names"]["value"].index('model')
        model_json = surrogate_json["value"][model_json_index]

        # Extract closure_lowest_avg_load_used
        closure_lowest_avg_load_used=closure_lowest_avg_load_used_default
        if "closure_lowest_avg_load_used" in surrogate_json["attributes"]["names"]["value"]:
            closure_lowest_avg_load_used_index = surrogate_json["attributes"]["names"]["value"].index('closure_lowest_avg_load_used')
            closure_lowest_avg_load_used=float(surrogate_json["value"][closure_lowest_avg_load_used_index]["value"][0])
            pass

        bendingstress_index = surrogate_json["attributes"]["names"]["value"].index("bendingstress")
        bendingstress = float(surrogate_json["value"][bendingstress_index]["value"][0])

        dynamicnormalstressampl_index = surrogate_json["attributes"]["names"]["value"].index("dynamicnormalstressampl")
        dynamicnormalstressampl = float(surrogate_json["value"][dynamicnormalstressampl_index]["value"][0])
        

        dynamicshearstressampl_index = surrogate_json["attributes"]["names"]["value"].index("dynamicshearstressampl")
        dynamicshearstressampl = float(surrogate_json["value"][dynamicshearstressampl_index]["value"][0])
        

        thermalpower_index = surrogate_json["attributes"]["names"]["value"].index("thermalpower")
        thermalpower_str = surrogate_json["value"][thermalpower_index]["value"][0]
        if thermalpower_str == "NA": # accommodate R NA values
            thermalpower = np.nan
            pass
        else:
            thermalpower = float(thermalpower_str)
            pass

        excfreq_index = surrogate_json["attributes"]["names"]["value"].index("excfreq")
        excfreq = float(surrogate_json["value"][excfreq_index]["value"][0])

        
        return  super(denormalized_surrogate,cls).fromjson(model_json,
                                                           params_nominal = params_nominal,
                                                           output_nominal = output_nominal,
                                                           closure_lowest_avg_load_used=closure_lowest_avg_load_used,
                                                           bendingstress=bendingstress,
                                                           dynamicnormalstressampl=dynamicnormalstressampl,
                                                           dynamicshearstressampl=dynamicshearstressampl,
                                                           thermalpower=thermalpower,
                                                           excfreq=excfreq,
                                                           **kwargs)
    pass



class nonnegative_denormalized_surrogate(denormalized_surrogate):
    # Class representing denormalized surrogate model
    # that operates on physical units and will not give negative answers


    def evaluate(self,new_positions,meanonly=False,accel_trisolve_devs=None):
        resdict = super(nonnegative_denormalized_surrogate,self).evaluate(new_positions,meanonly=meanonly,accel_trisolve_devs=accel_trisolve_devs)

        resdict["mean"][resdict["mean"] < 0.0] = 0.0  # Remove negative numbers from mean

        if not meanonly:
            resdict["lower95"][resdict["lower95"] < 0.0] = 0.0  # Remove negative numbers from lower95
            resdict["upper95"][resdict["upper95"] < 0.0] = 0.0  # Remove negative numbers from upper95
            pass
        
        return resdict

    def evaluate_derivative(self,new_positions,k,accel_trisolve_devs=None):

        # Need to evaluate function to find where it is negative
        evaldict = super(nonnegative_denormalized_surrogate,self).evaluate(new_positions,meanonly=True,accel_trisolve_devs=accel_trisolve_devs)

        negative_zone = evaldict["mean"] < 0.0

        derivative = super(nonnegative_denormalized_surrogate,self).evaluate_derivative(new_positions,k,accel_trisolve_devs=accel_trisolve_devs)

        # derivative is also zero in the negative_zone

        derivative[negative_zone] = 0.0
        
        return derivative

    
    @classmethod
    def fromjson(cls,surrogate_json,**kwargs):
        
        #surrogate_json = json.load(open('/tmp/test.json'))
        return  super(nonnegative_denormalized_surrogate,cls).fromjson(surrogate_json,**kwargs)
    pass

def load_denorm_surrogates_from_jsonfile(surrogates_jsonfile,nonneg=True,**kwargs):

    if nonneg:
        denormalized_surrogate_class = nonnegative_denormalized_surrogate
        pass
    else:
        denormalized_surrogate_class = denormalized_surrogate
        pass
    
    fh = open(surrogates_jsonfile)
    jsondata = json.load(fh)
    fh.close()

    denorm_surrogates = {}
    
    closure_lowest_avg_load_used_index = jsondata["attributes"]["names"]["value"].index('closure_lowest_avg_load_used')
    closure_lowest_avg_load_used=float(jsondata["value"][closure_lowest_avg_load_used_index]["value"][0])

    for attrindex in range(len(jsondata["attributes"]["names"]["value"])):
        attrname = jsondata["attributes"]["names"]["value"][attrindex]
        if attrname.startswith("bs_pa"):  # This is a surrogate, named according to dynamic normal stress amplitude, dynamic shear stress amplitude, and bending stress
            denorm_surrogates[attrname]=denormalized_surrogate_class.fromjson(jsondata["value"][attrindex],closure_lowest_avg_load_used_default = closure_lowest_avg_load_used, **kwargs)
            pass
        pass

    return denorm_surrogates
            

if __name__=="__main__":
    # obsolete test code fragments
    #new_positions = np.array((np.arange(0,1.0,.01),),dtype='d').T
    # new_positions = np.array(((30.2,0.0,10.0,.01),),dtype='d')


    from matplotlib import pyplot as pl
    pl.figure()
    pl.plot(new_positions,yhat,'-',
            new_positions,lower95,'-',
            new_positions,upper95,'-')
    pass
