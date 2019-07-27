# -*- coding: utf-8 -*-
"""
@author: hkaneko
"""

import math

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import model_selection, svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

dataset_flag = 3  # 0: linear simulation dataset 1, 1: linear simulation dataset 2, 2: nonlinear simulation dataset, 3: spectra dataset
transfer_learning_flag = 0  # 0: transfer learning, 1: using only target data, 2: using both supporting data and target data
regression_methods = ['gp']
#regression_methods = ['pls', 'rr', 'lasso', 'en', 'lsvr', 'nsvr', 'dt', 'rf', 'gp', 'lgb', 'xgb', 'gbdt']
#number_of_test_samples = 100
number_of_test_samples = 64

number_of_supporting_samples_in_simulation = 100
number_of_target_samples_in_simulation = 103
noise_ratio_in_simulation = 0.1
do_autoscaling = True  # True or False
threshold_of_rate_of_same_value = 0.99
fold_number = 5
max_pls_component_number = 30
ridge_lambdas = 2 ** np.arange(-5, 10, dtype=float)  # L2 weight in ridge regression
lasso_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # L1 weight in LASSO
elastic_net_lambdas = np.arange(0.01, 0.71, 0.01, dtype=float)  # Lambda in elastic net
elastic_net_alphas = np.arange(0.01, 1.00, 0.01, dtype=float)  # Alpha in elastic net
linear_svr_cs = 2 ** np.arange(-5, 5, dtype=float)  # C for linear svr
linear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for linear svr
nonlinear_svr_cs = 2 ** np.arange(-5, 10, dtype=float)  # C for nonlinear svr
nonlinear_svr_epsilons = 2 ** np.arange(-10, 0, dtype=float)  # Epsilon for nonlinear svr
nonlinear_svr_gammas = 2 ** np.arange(-20, 10, dtype=float)  # Gamma for nonlinear svr
random_forest_number_of_trees = 300  # Number of decision trees for random forest
random_forest_x_variables_rates = np.arange(1, 10,
                                            dtype=float) / 10  # Ratio of the number of X-variables for random forest
 
                                            
np.random.seed(0)                                                    
if dataset_flag == 0:
    x_supporting = np.random.rand(number_of_supporting_samples_in_simulation, 2)
    y_supporting = 2 * x_supporting[:, 0] + 3 * x_supporting[:, 1] + 1
    y_supporting = y_supporting + noise_ratio_in_simulation * y_supporting.std() * np.random.rand(len(y_supporting))
    x_target = np.random.rand(number_of_target_samples_in_simulation, 2)
    y_target = 2 * x_target[:, 0] + 3 * x_target[:, 1] + 2
    y_target = y_target + noise_ratio_in_simulation * y_target.std() * np.random.rand(len(y_target))
if dataset_flag == 1:
    x_supporting = np.random.rand(number_of_supporting_samples_in_simulation, 2)
    y_supporting = 2 * x_supporting[:, 0] + 3 * x_supporting[:, 1] + 1
    y_supporting = y_supporting + noise_ratio_in_simulation * y_supporting.std() * np.random.rand(len(y_supporting))
    x_target = np.random.rand(number_of_target_samples_in_simulation, 2)
    y_target = 2 * x_target[:, 0] + 4 * x_target[:, 1] + 1
    y_target = y_target + noise_ratio_in_simulation * y_target.std() * np.random.rand(len(y_target))
elif dataset_flag == 2:
    x_supporting = np.random.rand(number_of_supporting_samples_in_simulation, 2)
    y_supporting = 2 * (x_supporting[:, 0] - 2) ** 3 + 3 * x_supporting[:, 1] ** 2 + 1
    y_supporting = y_supporting + noise_ratio_in_simulation * y_supporting.std() * np.random.rand(len(y_supporting))
    x_target = np.random.rand(number_of_target_samples_in_simulation, 2)
    y_target = 2 * (x_target[:, 0] - 2) ** 3 + 3 * x_target[:, 1] ** 2 + 3
    y_target = y_target + noise_ratio_in_simulation * y_target.std() * np.random.rand(len(y_target))
elif dataset_flag == 3:
    # load data set
    raw_data_with_y_supporting = pd.read_csv('shootout_2012_pilot_scale.csv', encoding='SHIFT-JIS', index_col=0)
    raw_data_with_y_target = pd.read_csv('shootout_2012_full_scale.csv', encoding='SHIFT-JIS', index_col=0)
        
    raw_data_with_y_supporting_arr = np.array(raw_data_with_y_supporting)
    raw_data_with_y_target_arr = np.array(raw_data_with_y_target)
    
    y_supporting = raw_data_with_y_supporting_arr[:, 0]
    x_supporting = raw_data_with_y_supporting_arr[:, 1:]
    y_target = raw_data_with_y_target_arr[:, 0]
    x_target = raw_data_with_y_target_arr[:, 1:]
np.random.seed()
x_train_target, x_test_target, y_train_target, y_test = train_test_split(x_target, y_target, test_size=number_of_test_samples, random_state=0)

if transfer_learning_flag == 1:
    x_train = x_train_target.copy()
    x_test = x_test_target.copy()
    y_train = y_train_target.copy()
elif transfer_learning_flag == 2:
    x_train = np.r_[x_supporting, x_train_target]
    x_test = x_test_target.copy()
    y_train = np.r_[y_supporting, y_train_target]
elif transfer_learning_flag == 0:
    x_supporting_arranged = np.c_[x_supporting, x_supporting, np.zeros(x_supporting.shape)]
    x_train_target_arranged = np.c_[x_train_target, np.zeros(x_train_target.shape), x_train_target]
    x_train = np.r_[x_supporting_arranged, x_train_target_arranged]
    x_test = np.c_[x_test_target, np.zeros(x_test_target.shape), x_test_target]
    y_train = np.r_[y_supporting, y_train_target]

fold_number = min(fold_number, len(y_train))

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

# delete descriptors with high rate of the same values
rate_of_same_value = list()
num = 0
for X_variable_name in x_train.columns:
    num += 1
#    print('{0} / {1}'.format(num, x_train.shape[1]))
    same_value_number = x_train[X_variable_name].value_counts()
    rate_of_same_value.append(float(same_value_number[same_value_number.index[0]] / x_train.shape[0]))
deleting_variable_numbers = np.where(np.array(rate_of_same_value) >= threshold_of_rate_of_same_value)

"""
# delete descriptors with zero variance
deleting_variable_numbers = np.where( raw_Xtrain.var() == 0 )
"""

if len(deleting_variable_numbers[0]) != 0:
    x_train = x_train.drop(x_train.columns[deleting_variable_numbers], axis=1)
    x_test = x_test.drop(x_test.columns[deleting_variable_numbers], axis=1)
    print('Variable numbers zero variance: {0}'.format(deleting_variable_numbers[0] + 1))

print('# of X-variables: {0}'.format(x_train.shape[1]))

# autoscaling
if do_autoscaling:
    autoscaled_x_train = (x_train - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
    autoscaled_y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
    autoscaled_x_test = (x_test - x_train.mean(axis=0)) / x_train.std(axis=0, ddof=1)
else:
    autoscaled_x_train = x_train.copy()
    autoscaled_y_train = y_train.copy()
    autoscaled_x_test = x_test.copy()

plt.rcParams['font.size'] = 18  # 横軸や縦軸の名前の文字などのフォントのサイズ
for method in regression_methods:
    print(method)
    if method == 'pls':  # Partial Least Squares
        pls_components = np.arange(1, min(np.linalg.matrix_rank(autoscaled_x_train) + 1, max_pls_component_number + 1), 1)
        r2all = list()
        r2cvall = list()
        for pls_component in pls_components:
            pls_model_in_cv = PLSRegression(n_components=pls_component)
            pls_model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
            calculated_y_in_cv = np.ndarray.flatten(pls_model_in_cv.predict(autoscaled_x_train))
            estimated_y_in_cv = np.ndarray.flatten(
                model_selection.cross_val_predict(pls_model_in_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
            if do_autoscaling:
                calculated_y_in_cv = calculated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
    
            """
            plt.figure(figsize=figure.figaspect(1))
            plt.scatter( y, estimated_y_in_cv)
            plt.xlabel("Actual Y")
            plt.ylabel("Calculated Y")
            plt.show()
            """
            r2all.append(float(1 - sum((y_train - calculated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        plt.plot(pls_components, r2all, 'bo-')
        plt.plot(pls_components, r2cvall, 'ro-')
        plt.ylim(0, 1)
        plt.xlabel('Number of PLS components')
        plt.ylabel('r2(blue), r2cv(red)')
        plt.show()
        optimal_pls_component_number = np.where(r2cvall == np.max(r2cvall))
        optimal_pls_component_number = optimal_pls_component_number[0][0] + 1
        regression_model = PLSRegression(n_components=optimal_pls_component_number)
    elif method == 'rr':  # ridge regression
        r2cvall = list()
        for ridge_lambda in ridge_lambdas:
            rr_model_in_cv = Ridge(alpha=ridge_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(rr_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                  cv=fold_number)
            if do_autoscaling:
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        plt.figure()
        plt.plot(ridge_lambdas, r2cvall, 'k', linewidth=2)
        plt.xscale('log')
        plt.xlabel('Weight for ridge regression')
        plt.ylabel('r2cv for ridge regression')
        plt.show()
        optimal_ridge_lambda = ridge_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Ridge(alpha=optimal_ridge_lambda)
    elif method == 'lasso':  # LASSO
        r2cvall = list()
        for lasso_lambda in lasso_lambdas:
            lasso_model_in_cv = Lasso(alpha=lasso_lambda)
            estimated_y_in_cv = model_selection.cross_val_predict(lasso_model_in_cv, autoscaled_x_train, autoscaled_y_train,
                                                                  cv=fold_number)
            if do_autoscaling:
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            r2cvall.append(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2)))
        plt.figure()
        plt.plot(lasso_lambdas, r2cvall, 'k', linewidth=2)
        plt.xlabel('Weight for LASSO')
        plt.ylabel('r2cv for LASSO')
        plt.show()
        optimal_lasso_lambda = lasso_lambdas[np.where(r2cvall == np.max(r2cvall))[0][0]]
        regression_model = Lasso(alpha=optimal_lasso_lambda)
    elif method == 'en':  # Elastic net
        elastic_net_in_cv = ElasticNetCV(cv=fold_number, l1_ratio=elastic_net_lambdas, alphas=elastic_net_alphas)
        elastic_net_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_elastic_net_alpha = elastic_net_in_cv.alpha_
        optimal_elastic_net_lambda = elastic_net_in_cv.l1_ratio_
        regression_model = ElasticNet(l1_ratio=optimal_elastic_net_lambda, alpha=optimal_elastic_net_alpha)
    elif method == 'lsvr':  # Linear SVR
        linear_svr_in_cv = GridSearchCV(svm.SVR(kernel='linear'), {'C': linear_svr_cs, 'epsilon': linear_svr_epsilons},
                                        cv=fold_number)
        linear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_linear_svr_c = linear_svr_in_cv.best_params_['C']
        optimal_linear_svr_epsilon = linear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='linear', C=optimal_linear_svr_c, epsilon=optimal_linear_svr_epsilon)
    elif method == 'nsvr':  # Nonlinear SVR
        variance_of_gram_matrix = list()
        numpy_autoscaled_Xtrain = np.array(autoscaled_x_train)
        for nonlinear_svr_gamma in nonlinear_svr_gammas:
            gram_matrix = np.exp(
                -nonlinear_svr_gamma * ((numpy_autoscaled_Xtrain[:, np.newaxis] - numpy_autoscaled_Xtrain) ** 2).sum(
                    axis=2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_nonlinear_gamma = nonlinear_svr_gammas[
            np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        # CV による ε の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', C=3, gamma=optimal_nonlinear_gamma), {'epsilon': nonlinear_svr_epsilons},
                                   cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_epsilon = model_in_cv.best_params_['epsilon']
        # CV による C の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, gamma=optimal_nonlinear_gamma),
                                   {'C': nonlinear_svr_cs}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_c = model_in_cv.best_params_['C']
        # CV による γ の最適化
        model_in_cv = GridSearchCV(svm.SVR(kernel='rbf', epsilon=optimal_nonlinear_epsilon, C=optimal_nonlinear_c),
                                   {'gamma': nonlinear_svr_gammas}, cv=fold_number, iid=False, verbose=0)
        model_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_nonlinear_gamma = model_in_cv.best_params_['gamma']
#        nonlinear_svr_in_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_nonlinear_gamma),
#                                           {'C': nonlinear_svr_cs, 'epsilon': nonlinear_svr_epsilons}, cv=fold_number)
#        nonlinear_svr_in_cv.fit(autoscaled_x_train, autoscaled_y_train)
#        optimal_nonlinear_c = nonlinear_svr_in_cv.best_params_['C']
#        optimal_nonlinear_epsilon = nonlinear_svr_in_cv.best_params_['epsilon']
        regression_model = svm.SVR(kernel='rbf', C=optimal_nonlinear_c, epsilon=optimal_nonlinear_epsilon,
                                   gamma=optimal_nonlinear_gamma)
    elif method == 'rf':  # Random forest
        rmse_oob_all = list()
        for random_forest_x_variables_rate in random_forest_x_variables_rates:
            RandomForestResult = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
                max(math.ceil(x_train.shape[1] * random_forest_x_variables_rate), 1)), oob_score=True)
            RandomForestResult.fit(autoscaled_x_train, autoscaled_y_train)
            estimated_y_in_cv = RandomForestResult.oob_prediction_
            if do_autoscaling:
                estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
            rmse_oob_all.append((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)
        plt.figure()
        plt.plot(random_forest_x_variables_rates, rmse_oob_all, 'k', linewidth=2)
        plt.xlabel('Ratio of the number of X-variables')
        plt.ylabel('RMSE of OOB')
        plt.show()
        optimal_random_forest_x_variables_rate = random_forest_x_variables_rates[
            np.where(rmse_oob_all == np.min(rmse_oob_all))[0][0]]
        regression_model = RandomForestRegressor(n_estimators=random_forest_number_of_trees, max_features=int(
            max(math.ceil(x_train.shape[1] * optimal_random_forest_x_variables_rate), 1)), oob_score=True)
    elif method == 'gp':  # Gaussian process
        regression_model = GaussianProcessRegressor(ConstantKernel() * RBF() + WhiteKernel())
    elif method == 'lgb':  # LightGBM
        import lightgbm as lgb
    
        regression_model = lgb.LGBMRegressor()
    elif method == 'xgb':  # XGBoost
        import xgboost as xgb
    
        regression_model = xgb.XGBRegressor()
    elif method == 'gbdt':  # scikit-learn
        from sklearn.ensemble import GradientBoostingRegressor
    
        regression_model = GradientBoostingRegressor()
    regression_model.fit(autoscaled_x_train, autoscaled_y_train)
    
    # calculate y
    calculated_ytrain = np.ndarray.flatten(regression_model.predict(autoscaled_x_train))
    if do_autoscaling:
        calculated_ytrain = calculated_ytrain * y_train.std(ddof=1) + y_train.mean()
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y_train, calculated_ytrain, c='blue')
    y_max = np.max(np.array([np.array(y_train), calculated_ytrain]))
    y_min = np.min(np.array([np.array(y_train), calculated_ytrain]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Calculated Y')
    plt.show()
    # r2, RMSE, MAE
    print('r2: {0}'.format(float(1 - sum((y_train - calculated_ytrain) ** 2) / sum((y_train - y_train.mean()) ** 2))))
    print('RMSE: {0}'.format(float((sum((y_train - calculated_ytrain) ** 2) / len(y_train)) ** 0.5)))
    print('MAE: {0}'.format(float(sum(abs(y_train - calculated_ytrain)) / len(y_train))))
    
    # estimated_y in cross-validation
    estimated_y_in_cv = np.ndarray.flatten(
        model_selection.cross_val_predict(regression_model, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
    if do_autoscaling:
        estimated_y_in_cv = estimated_y_in_cv * y_train.std(ddof=1) + y_train.mean()
    # yy-plot
    plt.figure(figsize=figure.figaspect(1))
    plt.scatter(y_train, estimated_y_in_cv, c='blue')
    y_max = np.max(np.array([np.array(y_train), estimated_y_in_cv]))
    y_min = np.min(np.array([np.array(y_train), estimated_y_in_cv]))
    plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
             [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
    plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    plt.xlabel('Actual Y')
    plt.ylabel('Estimated Y in CV')
    plt.show()
    # r2cv, RMSEcv, MAEcv
    print('r2cv: {0}'.format(float(1 - sum((y_train - estimated_y_in_cv) ** 2) / sum((y_train - y_train.mean()) ** 2))))
    print('RMSEcv: {0}'.format(float((sum((y_train - estimated_y_in_cv) ** 2) / len(y_train)) ** 0.5)))
    print('MAEcv: {0}'.format(float(sum(abs(y_train - estimated_y_in_cv)) / len(y_train))))
    
    # standard regression coefficients
    # standard_regression_coefficients = regression_model.coef_
    # standard_regression_coefficients = pd.DataFrame(standard_regression_coefficients)
    # standard_regression_coefficients.index = Xtrain.columns
    # standard_regression_coefficients.columns = ['standard regression coefficient']
    # standard_regression_coefficients.to_csv( 'standard_regression_coefficients.csv' )
    
    # prediction
    if x_test.shape[0]:
        predicted_ytest = np.ndarray.flatten(regression_model.predict(autoscaled_x_test))
        if do_autoscaling:
            predicted_ytest = predicted_ytest * y_train.std(ddof=1) + y_train.mean()
        # yy-plot
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(y_test, predicted_ytest, c='blue')
        y_max = np.max(np.array([np.array(y_test), predicted_ytest]))
        y_min = np.min(np.array([np.array(y_test), predicted_ytest]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Predicted Y')
        plt.show()
        # r2p, RMSEp, MAEp
        print('r2p: {0}'.format(float(1 - sum((y_test - predicted_ytest) ** 2) / sum((y_test - y_test.mean()) ** 2))))
        print('RMSEp: {0}'.format(float((sum((y_test - predicted_ytest) ** 2) / len(y_test)) ** 0.5)))
        print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_ytest)) / len(y_test))))
        