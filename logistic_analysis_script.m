% SCRIPT TO USE A LOGISTIC MODEL TO ANALYSE CHOICE DATA

logistic_obj = logistic_analysis_obj();
logistic_obj.compute_condiff();
logistic_obj.abs_diff = logistic_obj.compute_abs(logistic_obj.con_diff);
logistic_obj.abs_diff = logistic_obj.compute_nanzscore(logistic_obj.abs_diff);
logistic_obj.compute_prevrew();
logistic_obj.compute_ru();
logistic_obj.cat_choice = logistic_obj.compute_cat(logistic_obj.data.correct);

% ADD RELEVANT REGRESSORS TO THE DATA
logistic_obj.add_vars(logistic_obj.prev_rew,'prev_rew');
logistic_obj.add_vars(logistic_obj.abs_diff,'abs_diff');
logistic_obj.add_vars(logistic_obj.ru,'reward_unc');
logistic_obj.add_vars(logistic_obj.cat_choice,'correct_cat');

% EXCLUDE TRIALS
logistic_obj.remove_conditions();

% FIT LOGISTIC MODEL
[betas_all,rsquared_all,h,p,t,coeffs_name] = logistic_obj.get_coeffs();