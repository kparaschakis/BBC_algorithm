# Load libraries
import numpy as np
from jadbio_internal.ml.hyperparam_conf import HyperparamConf
from jadbio_internal.ml.fs.LASSO import LASSO
from jadbio_internal.ml.fs.SESParams import SESParams
from jadbio_internal.ml.fs.Univariate import Univariate
from jadbio_internal.ml.algo.DT import DT
from jadbio_internal.ml.algo.dt.SplitFunction import SplitFunction
from jadbio_internal.ml.algo.dt.SplitCriterion import SplitCriterion
from jadbio_internal.ml.algo.RF import RF, Bagger, LossFunction
from jadbio_internal.ml.algo.SVM import LinearSVM, RBFSVM, PolynomialSVM
from jadbio_internal.ml.algo.Logistic import Logistic


def make_configuration(description):
    fs = None
    if '(SES)' in description:
        maxK = int(description.split('maxK = ')[1].split(', ')[0])
        ses_alpha = np.float(description.split('alpha = ')[1].split(' and budget ')[0])
        fs = SESParams(max_k=maxK, threshold=ses_alpha, max_vars=25)
    elif 'LASSO' in description:
        penalty = np.float(description.split('penalty=')[1].split(')')[0])
        fs = LASSO(max_vars=25, penalty=penalty)
    elif 'Univariate feature selection' in description:
        uni_alpha = np.float(description.split('alpha=')[1].split(',')[0])
        maxVars = int(description.split('maxVars=')[1].split(' |')[0])
        fs = Univariate(uni_alpha, maxVars)

    if 'Classification Decision Tree' in description:
        minLS = int(description.split('leaf size = ')[1].split(',')[0])
        dt_alpha = np.float(description.split('pruning parameter alpha = ')[1])
        ml = DT(mls=minLS, splits=1, alpha=dt_alpha, criterion=SplitCriterion.Deviance, vars_f=SplitFunction.sqrt(1))
    elif 'Random Forest' in description:
        minLS = int(description.split('leaf size = ')[1].split(',')[0])
        dt_vars_f = np.float(description.split('variables to split = ')[1].split(' sqrt')[0])
        rf_dt = DT(mls=minLS, splits=1, alpha=1, criterion=SplitCriterion.Deviance,
                   vars_f=SplitFunction.sqrt(dt_vars_f))
        n_trees = int(description.split('Random Forest training ')[1].split(' trees')[0])
        ml = RF(nmodels=n_trees, dt=rf_dt, loss=LossFunction.Deviance, bagger=Bagger.Probability)
    elif '(SVM)' in description:
        cost = np.float(description.split('cost = ')[1].split(',')[0])
        if 'Linear Kernel' in description:
            ml = LinearSVM(cost=cost)
        elif 'Polynomial Kernel' in description:
            gamma = np.float(description.split('gamma = ')[1].split(',')[0])
            degree = int(description.split('degree = ')[1])
            ml = PolynomialSVM(degrees=degree, cost=cost, gamma=gamma)
        else:
            gamma = np.float(description.split('gamma = ')[1])
            ml = RBFSVM(cost=cost, gamma=gamma)
    else:
        logistic_lambda = np.float(description.split('hyper-parameter lambda = ')[1].split(',')[0])
        ml = Logistic(l=logistic_lambda)
    best_configuration = HyperparamConf.with_default_pp(fs, ml)
    return best_configuration
