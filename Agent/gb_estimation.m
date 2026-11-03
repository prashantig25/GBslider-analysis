% """ This script runs the parameter estimation and the evaluation of the agent-based computational models
% 
%     1. Estimate sensory noise parameter sigma based on Experiment 3
%     2. Beta parameter of A1
%     3. Beta parameter of A2
%     4. Beta and lambda parameter of A3
%     5. Alpha parameter of A4
%     6. Alpha parameter of A5
%     7. Lambda and alpha parameter of A6
%     8. Evaluate models without estimation to plot the choice likelihoods
%     9. Save BIC as .csv for Bayesian model comparison
% """


% import numpy as np
% import pandas as pd
% import matplotlib.pyplot as plt
% from GbTaskVars import TaskVars
% from GbEstVars import EstVars
% from gb_parallelest import parallelest
% from GbEstimation import GbEstimation
% from gb_eval import evalfun

clc
clearvars

% Get data of pilot/main experiment
data = readtable("study1.txt");

% Number of subjects
num_subjs = length(unique(data.ID));

% Task parameters
task_vars = taskvars();

% Estimation parameters
est_vars = EstVars();

%  Estimation object
GbEstimation = GbEstimation();

% Initialize data frame for estimated parameters
params = NaN(num_subjs, 1); % Creates an N-by-1 array of NaN (float type)

% Set number of starting points and kernels for parallel estimation
est_vars.n_sp = 1;
est_vars.n_ker = 4;
est_vars.rand_sp = false;

% ---------------------------------------------------------------
% 1. Estimate sensory noise parameter sigma based on Experiment 3
% ---------------------------------------------------------------

% Adjust est_vars
est_vars.est_sigma = true;
est_vars.agent = 1;

% Set number of blocks for estimation
% est_vars.B = len(set(list(gb_data['b_t'])));
% est_vars.T = sum(gb_data[gb_data['id'] == 0]['b_t'] == 0);
est_vars.B = numel(unique(data.blocks)); 
% Sum the occurrences where 'id' is 0 and 'b_t' is 0
est_vars.T = sum(data.blocks(data.ID == 0) == 0);
data.condiff = abs(data.contrast_right-data.contrast_left)./2;
% Estimate sigma parameter
parallelest(data, est_vars);

% Add results to data frame
params.id = results_df.id;
params.sigma = results_df.minimum;

% Set fixed params. First used with sigma parameter for estimation of learning parameters. Later on used for evaluation
% of model to plot single-trial predictions
% fixed_params = pd.DataFrame(index=range(0, N), dtype='float')
% fixed_params['sigma'] = params['sigma']

% # ------------------------
% # 2. Beta parameter of A1
% # ------------------------
% 
% # Select Agent A1
% est_vars.agent = 1
% 
% # Turn-off sigma-parameter estimation
% est_vars.est_sigma = False
% 
% # Estimate parameter
% results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params)
% 
% # Add results to data frame
% params['beta_A1'] = results_df['minimum_beta']
% params['A1_llh'] = results_df['llh']
% params['A1_d_BIC'] = results_df['d_BIC']
% params['A1_a_BIC'] = results_df['a_BIC']
% 
% # ------------------------
% # 3. Beta parameter of A2
% # ------------------------
% 
% # Select Agent A2
% est_vars.agent = 2
% 
% # Estimate parameter
% results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params)
% 
% # Add results to data frame
% params['beta_A2'] = results_df['minimum_beta']
% params['A2_llh'] = results_df['llh']
% params['A2_d_BIC'] = results_df['d_BIC']
% params['A2_a_BIC'] = results_df['a_BIC']
% 
% # ----------------------------------
% # 4. Beta and lambda parameter of A3
% # ----------------------------------
% 
% # Select Agent A3
% est_vars.agent = 3
% 
% # Estimate parameter
% results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params)
% 
% # Add results to data frame
% params['lambda_A3'] = results_df['minimum_lambda']
% params['beta_A3'] = results_df['minimum_beta']
% params['A3_llh'] = results_df['llh']
% params['A3_d_BIC'] = results_df['d_BIC']
% params['A3_a_BIC'] = results_df['a_BIC']

% ------------------------
% 5. Alpha and beta parameter of A4
% ------------------------

% Select A4
est_vars.agent = 4;

% Estimate parameter
results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params);

% Add results to data frame
params.alpha_A4 = results_df.minimum_alpha;
params.beta_A4 = results_df.minimum_beta;
params.A4_llh = results_df.llh;
params.A4_d_BIC = results_df.d_BIC;
params.A4_a_BIC = results_df.a_BIC;

% # -------------------------
% # 6. Alpha parameter of A5
% # -------------------------
% 
% # Select A5
% est_vars.agent = 5
% 
% # Estimate parameter
% results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params)
% 
% # Add results to data frame
% params['alpha_A5'] = results_df['minimum_alpha']
% params['beta_A5'] = results_df['minimum_beta']
% params['A5_llh'] = results_df['llh']
% params['A5_d_BIC'] = results_df['d_BIC']
% params['A5_a_BIC'] = results_df['a_BIC']
% 
% # -----------------------------------
% # 7. Lambda and alpha parameter of A6
% # -----------------------------------
% 
% # Select A6
% est_vars.agent = 6
% 
% # Estimate parameters
% results_df = parallelest(gb_data, est_vars, fixed_params=fixed_params)
% 
% # Add results to data frame
% params['lambda_A6'] = results_df['minimum_lambda']
% params['alpha_A6'] = results_df['minimum_alpha']
% params['beta_A6'] = results_df['minimum_beta']
% params['A6_llh'] = results_df['llh']
% params['A6_d_BIC'] = results_df['d_BIC']
% params['A6_a_BIC'] = results_df['a_BIC']

% # --------------------------------------------------------------------
% # 8. Evaluate models without estimation to plot the choice likelihoods
% # --------------------------------------------------------------------
% 
% # Add parameters to fixed params for plotting the choice likelihoods
% % fixed_params['beta_A1'] = params['beta_A1']
% % fixed_params['beta_A2'] = params['beta_A2']
% % fixed_params['beta_A3'] = params['beta_A3']
% fixed_params['beta_A4'] = params['beta_A4']
% % fixed_params['beta_A5'] = params['beta_A5']
% % fixed_params['beta_A6'] = params['beta_A6']
% fixed_params['alpha_A4'] = params['alpha_A4']
% % fixed_params['alpha_A5'] = params['alpha_A5']
% % fixed_params['alpha_A6'] = params['alpha_A6']
% % fixed_params['lambda_A3'] = params['lambda_A3']
% % fixed_params['lambda_A6'] = params['lambda_A6']
% 
% # Evaluation
% df_bic = evalfun(gb_data, est_vars, fixed_params.copy(), save_plot=1)
% 
% # Add results to data frame
% params['A0_llh'] = df_bic.loc[df_bic['agent'] == 0, 'llh']
% params['A0_d_BIC'] = df_bic.loc[df_bic['agent'] == 0, 'd_BIC']
% params['A0_a_BIC'] = df_bic.loc[df_bic['agent'] == 0, 'a_BIC']
% 
% # Save parameters
% params.to_pickle('gb_data/modelbased.pkl')
% 
% # -------------------------------------------------
% # 9. Save BIC as .csv for Bayesian model comparison
% # -------------------------------------------------
% 
% part_bic_mat = np.full([N, 7], np.nan)
% part_bic_mat[:, 0] = params['A0_d_BIC'] + params['A0_a_BIC']
% part_bic_mat[:, 1] = params['A1_d_BIC'] + params['A1_a_BIC']
% part_bic_mat[:, 2] = params['A2_d_BIC'] + params['A2_a_BIC']
% part_bic_mat[:, 3] = params['A3_d_BIC'] + params['A3_a_BIC']
% part_bic_mat[:, 4] = params['A4_d_BIC'] + params['A4_a_BIC']x
% part_bic_mat[:, 5] = params['A5_d_BIC'] + params['A5_a_BIC']
% part_bic_mat[:, 6] = params['A6_d_BIC'] + params['A6_a_BIC']
% np.savetxt('gb_data/part_bic_mat.csv', part_bic_mat, delimiter=',')
% 
% # Plot estimates
% plt.show()
