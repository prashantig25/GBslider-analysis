%% MODEL SPACE
clc
clearvars 

model1 = 'up ~ pe + pe:contrast_diff + pe:salience + pe:congruence';
model2 = 'up ~ pe + pe:condition + pe:salience + pe:congruence';
model3 = 'up ~ pe + pe:contrast_diff + pe:salience +pe:congruence + pe:subj_est_unc + pe:pe_sign + pe:reward_unc';
model4 = 'up ~ pe + pe:contrast_diff + pe:condition:contrast_diff + pe:salience';
model5 = 'up ~ pe + pe:contrast_diff + pe:reward_unc +pe:reward_unc:contrast_diff + pe:salience +pe:congruence + pe:subj_est_unc + pe:pe_sign';
model6 = 'up ~ pe + pe:reward_unc + pe:condition:reward_unc + pe:salience';
model7 = 'up ~ pe + pe:reward_unc + pe:up_bs_diff+ pe:salience';
%% EXTRACT, RECODE AND COMPUTE ALL REGRESSORS

colors;
data_subjs = readtable("data_subjs.xlsx");

% FLIP THE REPORTED MU FOR INCONGRUENT BLOCKS
for i = 1:height(data_subjs)
    if data_subjs.congruence(i) == 0
        data_subjs.flipped_mu(i) = 1-data_subjs.mu(i);
    else
        data_subjs.flipped_mu(i) = data_subjs.mu(i);
    end
end

% COMPUTE THE MU FOR THE INCORRECT ACTION
for i = 1:height(data_subjs)
    if data_subjs.congruence(i) == 0
        data_subjs.incorr_mu(i) = data_subjs.mu(i);
    else
        data_subjs.incorr_mu(i) = 1-data_subjs.mu(i);
    end
end


% STORE MU DEPENDING ON THE OBSERVED REWARD
for i = 1:height(data_subjs)-1
     if data_subjs.choice_corr(i+1) == 1 
        data_subjs.actual_mu_corr(i) = data_subjs.flipped_mu(i);
     elseif data_subjs.choice_corr(i+1) == 0 
        data_subjs.actual_mu_corr(i) = 1- data_subjs.flipped_mu(i);
     end
end

% INITIALIZE TABLE COLUMNS FOR PE AND UP
data_subjs.pe_actual_corr = rand(height(data_subjs),1);
data_subjs.up_actual_corr = rand(height(data_subjs),1);

% COMPUTE PE AND UP
% PE = 
% UP = 
for t = 2:height(data_subjs)
        data_subjs.pe_actual_corr(t) = data_subjs.correct(t) - data_subjs.actual_mu_corr(t-1);
        if data_subjs.choice_corr(t) == 1
            data_subjs.up_actual_corr(t) = data_subjs.flipped_mu(t)-data_subjs.flipped_mu(t-1);
        else
            data_subjs.up_actual_corr(t) = data_subjs.incorr_mu(t)-data_subjs.incorr_mu(t-1);
        end
end

% SET PE AND UP FOR 1st TRIAL TO 0
data_subjs.pe_actual_corr(data_subjs.trials == 1) = 0;
data_subjs.up_actual_corr(data_subjs.trials == 1) = 0;

% COMPUTE VARIOUS REGRESSORS

% SUBJECTIVE ESTIMATION UNCERTAINTY
data_subjs.subj_est_unc = data_subjs.predicted_mu.*(1-data_subjs.predicted_mu);

% REWARD UNCERTAINTY
for i = 1:height(data_subjs)
    if data_subjs.choice_cond(i) == 2
        data_subjs.reward_unc(i) = 1;
    else
        data_subjs.reward_unc(i) = 0;
    end
end

% PERCEPTUAL UNCERTAINTY
for i = 1:height(data_subjs)
    if data_subjs.choice_cond(i) == 3
        data_subjs.pu(i) = 0;
    else
        data_subjs.pu(i) = 1;
    end
end

% PE SIGN
for i = 1:height(data_subjs)
    if data_subjs.pe_actual_corr(i) > 0
        data_subjs.pe_sign(i) = 1;
    else
        data_subjs.pe_sign(i) = 0;
    end
end

% NORMALIZING DATA FOR CONTINUOUS REGRESSORS
norm_data = NaN(height(data_subjs),1);
data_subjs.norm_subjest = normalise_zero_one(data_subjs.subj_est_unc,norm_data);
data_subjs.norm_condiff = normalise_zero_one(data_subjs.con_diff_choice,norm_data);

% GET RID OF REWARD CONDITION BEFORE FITTING THE MODEL 
all_cond = 0;
if all_cond == 0
    data_subjs = data_subjs(data_subjs.choice_cond ~= 3,:);
end

%% FITTING THE MODEL AND GETTING COEFFICIENTS

% SET VARIABLES TO RUN THE FUNCTION
id_subjs = unique(data_subjs.run_id);
mdl = model3; % which regression model
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','reward','pe_sign','pu'}; % cell array with names of categorical variables
num_vars = 7; % number of predictor vars
res_subjs = []; % empty array to store residuals
weight_y_n = 0; % non-weighted regression

% INITIALISE VARIABLES
int_subjs = NaN(99,1);
betas_all = NaN(99,num_vars);
rsquared_full = NaN(99,1);

% FIT THE MODEL TO GET RESIDUALS 
for i = 1:length(id_subjs)
    tbl = table(data_subjs.pe_actual_corr(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)), ...
        data_subjs.up_actual_corr(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)), ...
        round(data_subjs.norm_condiff(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),2), ...
        data_subjs.contrast(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.choice_cond(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.congruence(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.reward_unc(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.norm_subjest(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.recoded_rew(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.predicted_mu(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.pe_sign(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.pu(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
        ,'reward_unc','subj_est_unc','reward','mu','pe_sign','pu'});
    
    [betas,rsquared,residuals] = linear_fit(tbl,mdl,pred_vars,resp_var,cat_vars,num_vars,weight_y_n);
    res_subjs = [res_subjs; residuals, repelem(id_subjs(i),length(residuals)).'];
end

% WEIGHTED REGRESSION USING RESIDUALS
[wt_subjs] = weights(data_subjs, res_subjs);
wt_subjs(:,2) = res_subjs(:,2);
weight_y_n = 1;
for i = 1:length(id_subjs)
    weights_subj = wt_subjs(wt_subjs(:,2) == id_subjs(i));
    tbl = table(data_subjs.pe_actual_corr(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)), ...
        data_subjs.up_actual_corr(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)), ...
        round(data_subjs.norm_condiff(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),2), ...
        data_subjs.contrast(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.choice_cond(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.congruence(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.reward_unc(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.norm_subjest(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.recoded_rew(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.predicted_mu(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.pe_sign(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        data_subjs.pu(and(data_subjs.run_id == id_subjs(i),data_subjs.pe_actual_corr ~= 0)),...
        'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
        ,'reward_unc','subj_est_unc','reward','mu','pe_sign','pu'});    
    [betas,rsquared,residuals] = linear_fit(tbl,mdl,pred_vars,resp_var,cat_vars,num_vars,weight_y_n,weights_subj);
    betas_all(i,:) = betas(2:end);
    rsquared_full(i,1) = rsquared;
end

%% SIGNIFICANCE TESTING

% INITIALISE ARRAYS TO STORE h AND p VALUES
h_subjs = nan(1,num_vars);
p_subjs = nan(1,num_vars);
coeffs = betas_all; %[b1_agent,b2_agent,b3_agent,b4_agent,b5_agent,b6_agent,b7_agent,b8_agent];

% RUN TTESTS
for i = 1:num_vars
    [h_subjs(1,i),p_subjs(1,i)] = ttest(coeffs(:,i));
end