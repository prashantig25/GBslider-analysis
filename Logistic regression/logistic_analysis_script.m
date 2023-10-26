% SCRIPT TO USE A LOGISTIC MODEL TO ANALYSE CHOICE DATA
logistic_obj = logistic_analysis_obj(); % initialize logistic regression object
logistic_obj.compute_condiff(); % compute contrast difference for each trial
logistic_obj.abs_diff = logistic_obj.compute_abs(logistic_obj.con_diff); % compute absolute contrast difference values
logistic_obj.abs_diff = logistic_obj.compute_nanzscore(logistic_obj.abs_diff); % z-scored absolute contrast difference
logistic_obj.compute_prevrew(); % compute previous trial's reward
logistic_obj.compute_ru(); % compute reward uncertainty level
logistic_obj.cat_choice = logistic_obj.compute_cat(logistic_obj.data.correct); % transform choices to categorical variable

% ADD RELEVANT REGRESSORS TO THE DATA TABLE
logistic_obj.add_vars(logistic_obj.prev_rew,'prev_rew'); % previous trial's reward
logistic_obj.add_vars(logistic_obj.abs_diff,'abs_diff'); % absolute contrast difference
logistic_obj.add_vars(logistic_obj.ru,'reward_unc'); % reward uncertainty level
logistic_obj.add_vars(logistic_obj.cat_choice,'correct_cat'); % categorical choice

% EXCLUDE TRIALS
logistic_obj.remove_conditions(); % exclude conditions, if need be

% FIT LOGISTIC MODEL
[betas_all,rsquared_all,h,p,t,coeffs_name] = logistic_obj.get_coeffs();