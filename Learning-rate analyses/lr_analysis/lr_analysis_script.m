% SCRIPT TO RUN MODEL BASED ANALYSIS OF LEARNING RATES

preprocess_obj = preprocess(); % initialise object with all required variables and functions
preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_incorr_mu(); % compute reported contingency parameter, for the less rewarding option
preprocess_obj.compute_mu_pe(); % compute reported contingency parameter, for prediction errors
preprocess_obj.get_rew_mu(); % compute required variables for PE and UP

% COMPUTE VARS FOR LINEAR FIT
preprocess_obj.get_pe_up(); % predicition errors and updates 
preprocess_obj.compute_subjest; % subjective estimation uncertainty
preprocess_obj.compute_ru(); % reward uncertainty
preprocess_obj.compute_pesign(); % sign of prediction error
norm_subjest = preprocess_obj.compute_normalise(preprocess_obj.subjest); % normalised subjective estimation uncertainty
norm_condiff = preprocess_obj.compute_normalise(preprocess_obj.data.con_diff_choice); % normalised contrast difference

% ADD VARIABLES TO THE DATA TABLE
preprocess_obj.add_vars(norm_subjest,{'norm_subjest'});
preprocess_obj.add_vars(norm_condiff,{'norm_condiff'});
preprocess_obj.add_vars(preprocess_obj.ru,'reward_unc');
preprocess_obj.add_vars(preprocess_obj.pe_sign,'pe_sign');

% EXCLUDE TRIALS
preprocess_obj.remove_conditions(); % remove conditions
preprocess_obj.remove_zero_pe(); % remove trials with PE = 0

% SAVE PREPROCESSED FILE
writetable(preprocess_obj.data,'preprocessed_data.xlsx');

% FIT THE MODEL
lr_analysis = lr_analysis_obj();
[betas_all,rsquared_full,residuals_reg,coeffs_name,posterior_up_subjs] = lr_analysis.get_coeffs();

% SAVE DATA 
save("betas_signed","betas_all"); % save betas