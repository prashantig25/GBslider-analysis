% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

lr_analysis = lr_analysis_obj(); % initialise object with all required variables and functions
lr_analysis.flip_mu(); % compute reported contingency parameter, after correcting for congruence
lr_analysis.compute_incorr_mu(); % compute reported contingency parameter, for the less rewarding option
lr_analysis.compute_mu_pe(); % compute reported contingency parameter, for prediction errors
lr_analysis.get_rew_mu(); 

% COMPUTE VARS FOR LINEAR FIT
lr_analysis.get_pe_up(); % predicition errors and updates 
lr_analysis.compute_subjest; % subjective estimation uncertainty
lr_analysis.compute_ru(); % reward uncertainty
lr_analysis.compute_pesign(); % sign of prediction error
norm_subjest = lr_analysis.compute_normalise(lr_analysis.subjest); % normalised subjective estimation uncertainty
norm_condiff = lr_analysis.compute_normalise(lr_analysis.data.con_diff_choice); % normalised contrast difference

% ADD VARIABLES TO THE DATA TABLE
lr_analysis.add_vars(norm_subjest,{'norm_subjest'});
lr_analysis.add_vars(norm_condiff,{'norm_condiff'});
lr_analysis.add_vars(lr_analysis.ru,'reward_unc');
lr_analysis.add_vars(lr_analysis.pe_sign,'pe_sign');

% EXCLUDE TRIALS
lr_analysis.remove_conditions();
lr_analysis.remove_zero_pe();

% FIT THE MODEL
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs();