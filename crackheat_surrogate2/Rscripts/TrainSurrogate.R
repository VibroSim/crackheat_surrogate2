# /usr/bin/env Rscript

# Debugging Process: 
# train_crackheat_surrogate processtrak step 
# prints out args= ... when run. 
# To debug, start R manually in a terminal, and paste
# that assigment. 
#
# Then you can run most of this script line-by-line
# 
# ... do NOT paste in the args = commandArgs line below. 
# 
# If you need to trace into training_eval() (Python code) 
# than do NOT paste in the py_multiprocessing  import or 
# the lines immediately below. 
# Instead paste in py$multiprocessing_pool = NULL... 
# This will disable parallel processing on the Python side
# making pdb escapes, etc. work. 


# requires DiceKriging package (run install.packages()
library('DiceKriging')

# requires jsonlite for storing output
library('jsonlite')

# we model crack heat power per cycle...
# This model applies to a particular crack

# physical inputs:
#   * Bending load
#   * Dynamic strain
# model parameters (must be assumed/explored):
#   * m*sqrt(R)
#   * friction coefficient   # NOTE: should add support to angled_friction_model for broadcasting over friction coefficients

# Other crack dependent parameters which are invariant
#   * material/material properties
#   * Crack tortuosity
#   * Crack closure state

args = commandArgs(trailingOnly=TRUE)

# First command line parameter, angular crack tortuosity standard deviation in degrees
tortuosity = as.numeric(args[1])
# second command line parameter: left side closure csv
leftclosure = args[2]
# Third command line parameter: right side closure csv
rightclosure = args[3]

# Fourth command line parameter: crackheat table csv
crackheat_table_csv = args[4]

closure_lowest_avg_load_used = args[5]

aleft = as.numeric(args[6])
aright = as.numeric(args[7])

sigma_yield = as.numeric(args[8])
tau_yield = as.numeric(args[9])

crack_model_normal_type = args[10]
crack_model_shear_type = args[11]
Symmetric_COD=as.logical(args[12])

E=as.numeric(args[13])
nu=as.numeric(args[14])
numdraws = strtoi(args[15],base = 10)

num_mus = strtoi(args[16],base = 10)
num_msqrtRs = strtoi(args[17],base = 10)
output_filename = args[18]

		
# Requires reticulate package for letting R call python

library('reticulate') 

# eighteenth (optional) command line parameter, python binary to use
if (length(args) >= 19) {
  python_path = args[19]
  use_python(python_path)
}

py = import_main()

py_sys=import('sys')

# Fixed parameter ranges:
#mu = seq(0.01,3.0,length=5)
#msqrtR = seq(500000,50e6,length=5) # m*sqrtR (asperities*sqrt(m)/m^2

# ***!!! As we change the ranges, should also change min_vals and max_vals in pt_steps/eval_crackheat_surrogate to match!***
log_mu = seq(log(0.01),log(2.0),length=num_mus)
log_msqrtR = seq(9.2,18.4,length=num_msqrtRs) # log(m*sqrtR) log(asperities*sqrt(m)/m^2

# Covariance values indicating region of significance
# for each sample.... for now just use our grid step
log_mu_cov = (log_mu[2]-log_mu[1])*2.0
log_msqrtR_cov = (log_msqrtR[2]-log_msqrtR[1])*2.0

heating_response_stddev = 2.5e-7 # assumed variability in temperature output (W/m^2/Hz) (was 1e-6)

# Nominal values of physical quantities, for normalization
log_mu_nominal=abs(log(0.1)) # nominal values should be positive
log_msqrtR_nominal = log(4e6)

heating_response_nominal = 1e-6


# identify (DynamicNormalStressAmpl, DynamicShearStressAmpl, BendingStress)
# string tuples from crackheat csv, so we can train one surrogate for each. 
crackheat_table = read.csv(crackheat_table_csv,colClasses=c("DynamicNormalStressAmpl..Pa."="character",
		 			  "DynamicShearStressAmpl..Pa."="character",
                                          "BendingStress..Pa."="character"),
					  stringsAsFactors=FALSE)
lookup_keys=paste("bs_pa",crackheat_table[["BendingStress..Pa."]],"dnsa_pa",crackheat_table[["DynamicNormalStressAmpl..Pa."]],"dssa_pa",crackheat_table[["DynamicShearStressAmpl..Pa."]],sep='_')
DynamicNormalStressAmpl = crackheat_table["DynamicNormalStressAmpl..Pa."]
DynamicShearStressAmpl = crackheat_table["DynamicShearStressAmpl..Pa."]
BendingStress = crackheat_table["BendingStress..Pa."]

# ThermalPower extracted only for manual comparison purposes in eval_crackheat_surrogate; not otherwise used
ThermalPower = crackheat_table["ThermalPower..W."]
ExcFreq = crackheat_table["ExcFreq..Hz."]

log_msqrtR_norm = log_msqrtR/log_msqrtR_nominal
log_mu_norm = log_mu/log_mu_nominal


testgrid = expand.grid(log_mu = log_mu, log_msqrtR=log_msqrtR) # ALWAYS use mu as first column so Python side can vectorize over mu

testgrid_norm = expand.grid(log_mu_norm=log_mu_norm, log_msqrtR_norm=log_msqrtR_norm) # ALWAYS use mu as first column so Python side can vectorize over mu

#py$testgrid = testgrid
#py$tortuosity = tortuosity
#py$leftclosure = leftclosure
#py$rightclosure = rightclosure
#repl_python()



py_multiprocessing = import("multiprocessing")
if (py_sys$version_info$major >= 3) {
  py_multiprocessing$set_start_method('spawn')
}
py$multiprocessing_pool = py_multiprocessing$Pool()
#py$multiprocessing_pool = NULL

# import crackheat_surrogate2 package
crackheat_surrogate2 = import("crackheat_surrogate2")


all_surrogates = list()

all_surrogates$closure_lowest_avg_load_used = closure_lowest_avg_load_used

# lookup_keys lists the various measurements by bending stress
# and dynsmic normal stress amplitude
for (rownum in 1:NROW(lookup_keys)) {

    # Data structure to store our model data
    modeldata = list()

    modeldata$params_nominal = c(log_mu_nominal,log_msqrtR_nominal)
    modeldata$output_nominal = heating_response_nominal

    modeldata$bendingstress = as.numeric(BendingStress[rownum,])
    modeldata$dynamicnormalstressampl = as.numeric(DynamicNormalStressAmpl[rownum,])
    modeldata$dynamicshearstressampl = as.numeric(DynamicShearStressAmpl[rownum,])

    modeldata$thermalpower = as.numeric(ThermalPower[rownum,])
    modeldata$excfreq = as.numeric(ExcFreq[rownum,])
    
    #print(paste('rownum=',rownum))   

    max_normalized_stddev = 0.5
    max_unnormalized_stddev = max_normalized_stddev*heating_response_nominal

    training_eval_output = crackheat_surrogate2$training_eval$training_eval(testgrid,modeldata$bendingstress,modeldata$dynamicnormalstressampl,modeldata$dynamicshearstressampl,tortuosity,leftclosure,rightclosure,aleft,aright,sigma_yield,tau_yield,crack_model_normal_type,crack_model_shear_type,E,nu,numdraws,max_stddev = max_unnormalized_stddev,multiprocessing_pool=py$multiprocessing_pool,Symmetric_COD=Symmetric_COD)
    heating_response = training_eval_output[[1]]
    noise.stddev = training_eval_output[[2]]
    soft_closure_diags = training_eval_output[[3]]

    heating_response_norm = heating_response/heating_response_nominal

    noise.stddev_norm = noise.stddev/heating_response_nominal
    noise.variance_norm = noise.stddev_norm^2


    # For more info on DiceKriging, see
    # DiceKriging, DiceOptim: Two R Packages for the
    # Analysis of Computer Experiments by
    # Kriging-Based Metamodeling and Optimization
    #  In the Journal of Statistical Software (2012)

    # With universal kriging (UK) we represent the function
    # as a polynomial plus a deviation.
    # coef.var represents prior estimate of the variance of the deviation
    # coef.cov represents characteristic distance of each
    #          parameter 
    # noise.var parameter should come from measured variance at each
    #           point in the testing grid

    # use coef.cov parameter to compensate magnitude differences between parameters
    # (same units as parameter)... unnecessary now that they are normalized

    ## coef.var represents confidence of output result
    #coef_var = (100e-3)^2 
    #coef_cov = c(30e6,10e6,4e6,0.1)
    coef_var = (heating_response_stddev/heating_response_nominal)^2
    coef_cov = c(log_mu_cov/log_mu_nominal,log_msqrtR_cov/log_msqrtR_nominal)
    
    # General formula for the trend, but not including coefficients
    formula= ~log_mu_norm + log_msqrtR_norm + I(log_mu_norm^2) + I(log_msqrtR_norm^2) +  I(log_mu_norm*log_msqrtR_norm) 

    modeldata$model = km(formula=formula,design = data.frame(testgrid_norm), response = heating_response_norm, noise.var = noise.variance_norm, coef.var = coef_var, coef.cov=coef_cov)

    modeldata$model

    # add modeldata into all_surrogates
    all_surrogates[[lookup_keys[[rownum]]]] = modeldata
    ## p <- predict.km(modeldata$model, newdata = data.frame(x = t), type = "UK")
    # mu_test_vals = seq(from=.02,to=1.0,by=.01)
    # log_mu_test_vals = log(mu_test_vals)
    # log_msqrtR_test_val = testgrid[67,2]
    # 
    # p <- predict.km(modeldata$model, newdata = data.frame(log_mu_norm = log_mu_test_vals/log_mu_nominal, log_msqrtR_norm = log_msqrtR_test_val/log_msqrtR_nominal), type="UK")
    # plot(testgrid[67:77,1],heating_response_norm[67:77])
    # lines(log_mu_test_vals,p$mean)
}

write(serializeJSON(all_surrogates,digits=NA),file=output_filename)


# Verify that R and Python reconstruction code give same result
py_surrogates = crackheat_surrogate2$load_surrogate$load_denorm_surrogates_from_jsonfile(output_filename,nonneg=FALSE)

test_log_mu = log(0.3)
test_log_msqrtR = log(5e6)

test_log_mu_norm = test_log_mu/log_mu_nominal
test_log_msqrtR_norm = test_log_msqrtR/log_msqrtR_nominal

#test_positions_py = t(matrix(c(test_log_mu,test_log_msqrtR))) # create single-row matrix
test_positions_py = data.frame(log_mu = test_log_mu, log_msqrtR = test_log_msqrtR)

for (rownum in 1:NROW(lookup_keys)) {
    py_output = py_surrogates[[lookup_keys[[rownum]]]]$evaluate(test_positions_py)
    py_mean = py_output$mean[1]

    R_output = predict.km(all_surrogates[[lookup_keys[[rownum]]]][["model"]], newdata=data.frame(expand.grid(log_mu_norm = test_log_mu_norm, log_msqrtR_norm=test_log_msqrtR_norm)),type="UK")
    R_mean = R_output$mean[1]*heating_response_nominal

    # Check that R and Python libraries give the same result
    if (abs((py_mean-R_mean)/R_mean) > 1e-5) {
        print("SURROGATE ERROR: R and Python library reconstruction values disagree!")
	print("Python:")
  	print(py_mean)
  	print("R:")
  	print(R_mean)
  	quit(save="no",status=1)
    }
}





## Debugging: Save dataframe
#testgrid_data_frame = data.frame(testgrid)
#testgrid_data_frame["output"]=heating_response
#write.csv(testgrid_data_frame,"/tmp/TrainSurrogate_R_Dataframe.csv",row.names=FALSE)



### to read JSON file:
#fh=file('/tmp/test.json')
#jsontmp = fromJSON(readLines(fh),simplifyVector=FALSE)
#jsontmp$attributes$model$attributes$call = NULL # address problem #1, below
#modeldata=unserializeJSON(toJSON(jsontmp,auto_unbox=TRUE,digits=NA)) # two problems: 1. model@call doesn't transfer properly but can be safely removed. 2. model@trend.formula transfers but is inoperable (needs to be reparsed)
## fixup model@trend.formula... (address problem #2)
#modeldata$model@trend.formula = eval(parse(text=paste(as.character(modeldata$model@trend.formula),collapse='')))
