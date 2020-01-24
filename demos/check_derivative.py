# This script loads a surrogate .json file and compares
# the analytic derivative evaluated in load_surrogate()
# with a numerical derivative to verify 

import numpy as np
import pandas as pd

from crackheat_surrogate2 import load_surrogate


json_surrogates_file="/tmp/C18-AFVT-019H_surrogate.json"

surrogates = load_surrogate.load_denorm_surrogates_from_jsonfile(json_surrogates_file)

# Coordinates of the evaluation
mu = 0.3
log_msqrtR = np.log(5e6) # asperities*sqrt(m)/m^2

dmu = 0.005
dlog_msqrtR=.01

new_positions = pd.DataFrame(columns=["mu","bendingstress","dynamicstress","log_msqrtR"])
new_positions=new_positions.append({ # Eval coords
    "mu": mu,
    "log_msqrtR": log_msqrtR
    },ignore_index=True)

new_positions=new_positions.append({ # Eval coords + mu shift
    "mu": mu+dmu,
    "log_msqrtR": log_msqrtR
    },ignore_index=True)

new_positions=new_positions.append({ # Eval coords + log_msqrtR shift
    "mu": mu,
    "log_msqrtR": log_msqrtR + dlog_msqrtR
    },ignore_index=True)


for surrogate_key in surrogates:
    eval_output = surrogates[surrogate_key].evaluate(new_positions)
    yhat = eval_output["mean"]


    dyhat_dmu_numerical = (yhat[1]-yhat[0])/dmu
    
    dyhat_dlog_msqrtR_numerical = (yhat[2]-yhat[0])/dlog_msqrtR
    
    dyhat_dmu_symbolic = surrogates[surrogate_key].evaluate_derivative(new_positions,"mu")[0]  # only use result @ eval_coords
    dyhat_dlog_msqrtR_symbolic = surrogates[surrogate_key].evaluate_derivative(new_positions,"log_msqrtR")[0]

    print("%s dyhat_dmu_numerical: %g; dyhat_mu_symbolic: %g (should match)" % (surrogate_key,dyhat_dmu_numerical,dyhat_dmu_symbolic))

    print("%s dyhat_dlog_msqrtR_numerical: %g; dyhat_log_msqrtR_symbolic: %g (should match)" % (surrogate_key,dyhat_dlog_msqrtR_numerical,dyhat_dlog_msqrtR_symbolic))
    
    if np.abs((dyhat_dmu_numerical-dyhat_dmu_symbolic)/dyhat_dmu_symbolic) > .05:
        raise ValueError("%s mu derivative mismatch" % (surrogate_key))

    if np.abs((dyhat_dlog_msqrtR_numerical-dyhat_dlog_msqrtR_symbolic)/dyhat_dlog_msqrtR_symbolic) > .05:
        raise ValueError("%s log_msqrtR derivative mismatch" % (surrogate_key))
    pass
