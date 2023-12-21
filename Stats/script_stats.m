%%%%%% RUN ALL SORTS OF SIGNIFICANCE TESTS AND SAVE OUTPUT FOR MANUSCRIPT 
%%%%%% AND FIGURES

% INITIALISE VARS
clc
clearvars
save_descriptive = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\descriptive";
save_lr = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\lr_analysis";
save_integration = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\integration";
save_dm_bias = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\dm_analysis\bias";
save_dm_reg = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\dm_analysis\regression";
save_lr_suppl = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\" + ...
    "overleaf_folder_organisation\stats\supplement\learning";
%% LOAD DATA

% descriptive choice data
load("mix_salience_study2.mat","ecoperf_mix");
load("perc_salience_study2.mat","ecoperf_perc");
load("rew_salience_study2.mat","ecoperf_rew");

% descriptive learning data
load("mix_curves_study2.mat","mix_curve");
load("perc_curves_study2.mat","perc_curve");
load("rew_curves_study2.mat","rew_curve");

% LR analysis
betas_signed = load("betas_signed.mat","betas_all"); % signed 
betas_abs = load("betas_abs.mat","betas_all"); % absolute
load("lr_data.mat","data_subjs"); % preprocessed data

% integration results
load("lm_integration.mat","lm");

% salience bias
load("salience_bias.mat","salience_bias");
%% DESCRIPTIVE ANALYSIS FROM STUDY 2

% comparison to chance performance
chance_level = 0.5; % Specify the chance level (e.g., 0.5 for a binary task)

ecoperf_mix_avg = nanmean(ecoperf_mix,2);
ecoperf_perc_avg = nanmean(ecoperf_perc,2);
ecoperf_rew_avg = nanmean(ecoperf_rew,2);

% initialise
h_vals = NaN(3,2);
p_vals = NaN(3,2);
t_vals = NaN(3,1);
df_vals = NaN(3,1);
mean_ecoperf = NaN(3,1);
sem_ecoperf = NaN(3,1);
cond_names = ["mix";"perc";"rew"];

% Perform a one-sample t-test
[h_vals(1,:), p_vals(1,:), ~, stats_mix] = ttest(ecoperf_mix_avg, chance_level); 
[h_vals(2,:), p_vals(2,:), ~, stats_perc] = ttest(ecoperf_perc_avg, chance_level);
[h_vals(3,:), p_vals(3,:), ~, stats_rew] = ttest(ecoperf_rew_avg, chance_level);

t_vals(1,:) = stats_mix.tstat;
t_vals(2,:) = stats_perc.tstat;
t_vals(3,:) = stats_rew.tstat;

df_vals(1,:) = stats_mix.df;
df_vals(2,:) = stats_perc.df;
df_vals(3,:) = stats_rew.df;

[mean_ecoperf(1,:),sem_ecoperf(1,:)] = compute_mean_sem(ecoperf_mix_avg);
[mean_ecoperf(2,:),sem_ecoperf(2,:)] = compute_mean_sem(ecoperf_perc_avg);
[mean_ecoperf(3,:),sem_ecoperf(3,:)] = compute_mean_sem(ecoperf_rew_avg);

% save output to .csv file for OVERLEAF
ttestResults = table(cond_names, round(h_vals(:,1),2), round(p_vals(:,1),2), ...
    round(t_vals,2), round(df_vals,2), round(mean_ecoperf,2), round(sem_ecoperf,3),...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df','mean','sem'}); % Create a table to store the t-test results

% Save the table as a CSV file
writetable(ttestResults, strcat(save_descriptive,'\allconditions_chance_ttest.csv'));

%% ANOVA to compare three conditions

% initialise
mix_avg = nanmean(ecoperf_mix,2);
perc_avg = nanmean(ecoperf_perc,2);
rew_avg = nanmean(ecoperf_rew,2);

% anova
mean_avg = [mix_avg, perc_avg, rew_avg];
[~,tbl,~] = anova1(mean_avg);

% Save the table as a CSV file
name = "anova";
anova_results = table(name, round(tbl{2,5},2), round(tbl{2,6},2), tbl{2,3}, tbl{3,3}, ...
    'VariableNames', {'test', 'fvalue', 'pvalue', 'df1', 'df2'}); % Create a table to store the t-test results
writetable(anova_results, strcat(save_descriptive,'\ecoperf_anova.csv'));
%% T-TEST TO COMPARE THE IMPACT OF UNCERTAINTY ON CHOICES

% initialise
h_vals = NaN(2,1);
p_vals = NaN(2,1);
t_vals = NaN(2,1);
df_vals = NaN(2,1);
mean_ecoperf = NaN(3,1);
sem_ecoperf = NaN(3,1);

% t-test
[h_vals(1,:), p_vals(1,:), ~, stats_mixperc] = ttest2(mix_avg, perc_avg); % impact of reward uncertainty
[h_vals(2,:), p_vals(2,:), ~, stats_mixrew] = ttest2(mix_avg, rew_avg); % impact of perceptual uncertainty
cond_names = ["mix_perc";"mix_rew"];

t_vals(1,:) = stats_mixperc.tstat;
t_vals(2,:) = stats_mixrew.tstat;

df_vals(1,:) = stats_mixperc.df;
df_vals(2,:) = stats_mixrew.df;

[mean_ecoperf(1,:),sem_ecoperf(1,:)] = compute_mean_sem(mix_avg);
[mean_ecoperf(2,:),sem_ecoperf(2,:)] = compute_mean_sem(perc_avg);
[mean_ecoperf(3,:),sem_ecoperf(3,:)] = compute_mean_sem(rew_avg);

% save results
ttestResults = table(cond_names, round(h_vals(:,1),2), round(p_vals(:,1),2), ...
    round(t_vals,2), round(df_vals,2),...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df'}); 
writetable(ttestResults, strcat(save_descriptive,'\uncertainty_ttest.csv'));
%% ANOVA TO COMPARE SLIDER DATA ACROSS CONDITIONS

% initialise
mix_avg = nanmean(mix_curve,2);
perc_avg = nanmean(perc_curve,2);
rew_avg = nanmean(rew_curve,2);
mean_avg = [mix_avg, perc_avg, rew_avg];

% anova
[~,tbl,~] = anova1(mean_avg);

% Save the table as a CSV file
name = "anova";
anova_results = table(name, round(tbl{2,5},2), round(tbl{2,6},2), tbl{2,3}, tbl{3,3}, ...
    'VariableNames', {'test', 'fvalue', 'pvalue', 'df1', 'df2'}); % Create a table to store the t-test results
writetable(anova_results, strcat(save_descriptive,'\mu_anova.csv'));
%% T-TEST TO COMPARE SLIDER DATA ACROSS UNCERTAINTIES

% initialise
h_vals = NaN(2,1);
p_vals = NaN(2,1);
t_vals = NaN(2,1);
df_vals = NaN(2,1);

% t-test
[h_vals(1,:), p_vals(1,:), ~, stats_mixperc] = ttest2(mix_avg, perc_avg); % impact of reward uncertainty
[h_vals(2,:), p_vals(2,:), ~, stats_mixrew] = ttest2(mix_avg, rew_avg); % impact of perceptual uncertainty
cond_names = ["mix_perc";"mix_rew"];

t_vals(1,:) = stats_mixperc.tstat;
t_vals(2,:) = stats_mixrew.tstat;

df_vals(1,:) = stats_mixperc.df;
df_vals(2,:) = stats_mixrew.df;

% save
ttestResults = table(cond_names, round(h_vals(:,1),2), round(p_vals(:,1),2), round(t_vals,2), round(df_vals,2), ...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df'}); % Create a table to store the t-test results
writetable(ttestResults, strcat(save_descriptive,'\mu_uncertainty_ttest.csv'));
%% T-TEST ON BETA COEFFICIENTS

% t-test
cond_names = ["pe";"pe_condiff";"pe_salience";"pe_congruence";"pe_rewunc";"pe_estunc";"pe_pesign"];
[h_vals, p_vals, ~, stats] = ttest(betas_signed.betas_all); % Perform a one-sample t-test
t_vals = stats.tstat;
df_vals = stats.df;

% compute mean and SEM
mean_ecoperf = NaN(size(betas_signed.betas_all,2),1);
sem_ecoperf = NaN(size(betas_signed.betas_all,2),1);
for i = 1:size(betas_signed.betas_all,2)
    [mean_ecoperf(i,:),sem_ecoperf(i,:)] = compute_mean_sem(betas_signed.betas_all(:,i));
end

% save output to .csv file for OVERLEAF
ttestResults = table(cond_names, round(h_vals,2).', round(p_vals,4).', ...
    round(t_vals,2).', round(df_vals,2).', round(mean_ecoperf,2), round(sem_ecoperf,3),...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df','mean','sem'}); % Create a table to store the t-test results
writetable(ttestResults, strcat(save_lr,'\lr_betas_ttest.csv'));
%% COHEN'S D FOR BETA COEFFICIENTS

num_vars = size(betas_all,2);
cohen_d = NaN(num_vars,1);
cond_names = ["pe";"pe_condiff";"pe_salience";"pe_congruence";"pe_rewunc";"pe_estunc";"pe_pesign"];
for i = 1:num_vars
    cohen_d(i,1) = compute_cohen_ttest(nanmean(betas_all(:,i)),0,nanstd(betas_all(:,i)));
end

% save output to .csv file for OVERLEAF
cohenResults = table(cond_names, round(cohen_d,2), ...
    'VariableNames', {'Condition', 'cohend'}); % Create a table to store the t-test results
writetable(cohenResults, strcat(save_lr,'\lr_betas_cohen.csv'));
%% STATS FOR INTEGRATION RESULTS

% initialise
p_vals = lm.Coefficients.pValue;
cond_names = ["intercept";"pe";"pe_condiff";"pe_pesign"];

% save output to .csv file for OVERLEAF
cohenResults = table(cond_names, round(p_vals,4), ...
    'VariableNames', {'Condition', 'pval'}); % Create a table to store the t-test results
writetable(cohenResults, strcat(save_integration,'\integration_pvals.csv'));
%% SALIENCE BIAS STATS

% bias
bias_mix = ecoperf_mix(:,2) - ecoperf_mix(:,1);
bias_perc = ecoperf_perc(:,2) - ecoperf_perc(:,1);
bias_rew = ecoperf_rew(:,2) - ecoperf_rew(:,1);

% initialise
bias = [bias_mix, bias_perc, bias_rew];
h_vals = NaN(3,1);
p_vals = NaN(3,1);
t_vals = NaN(3,1);
df_vals = NaN(3,1);
bias_mean = NaN(3,1);
bias_sem = NaN(3,1);
cond_names = ["mix";"perc";"rew"];

% t-test
for c = 1:3
    [h_vals(c,:), p_vals(c,:), ~, stats] = ttest(bias(:,c));
    t_vals(c,:) = stats.tstat;
    df_vals(c,:) = stats.df;
    [bias_mean(c,:),bias_sem(c,:)] = compute_mean_sem(bias(:,c));
end

% save output to .csv file for OVERLEAF
ttestResults = table(cond_names, round(h_vals,2), round(p_vals,2), round(t_vals,2), ...
    round(df_vals,2), round(bias_mean,2), round(bias_sem,3),...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df','mean','sem'});
writetable(ttestResults, strcat(save_dm_bias,'\bias_ttest.csv'));
%% BIAS COMPARED ACROSS CONDITIONS
% anove
[p,tbl,~] = anova1(bias);

% Save the table as a CSV file
name = "anova";
anova_results = table(name, round(tbl{2,5},2), round(tbl{2,6},2), tbl{2,3}, tbl{3,3}, ...
    'VariableNames', {'test', 'fvalue', 'pvalue', 'df1', 'df2'}); 
writetable(anova_results, strcat(save_dm_bias,'\bias_anova.csv'));
%% BIAS COMPARED ACROSS UNCERTAINTIES
% initialise
h_vals = NaN(2,1);
p_vals = NaN(2,1);
t_vals = NaN(2,1);
df_vals = NaN(2,1);

% t-test
[h_vals(1,:), p_vals(1,:), ~, stats_mixperc] = ttest2(bias_mix, bias_perc); % impact of reward uncertainty
[h_vals(2,:), p_vals(2,:), ~, stats_mixrew] = ttest2(bias_mix, bias_rew); % impact of perceptual uncertainty
cond_names = ["mix_perc";"mix_rew"];

t_vals(1,:) = stats_mixperc.tstat;
t_vals(2,:) = stats_mixrew.tstat;

df_vals(1,:) = stats_mixperc.df;
df_vals(2,:) = stats_mixrew.df;

% save
ttestResults = table(cond_names, round(h_vals(:,1),2), round(p_vals(:,1),2), round(t_vals,2), round(df_vals,2), ...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df'}); 
writetable(ttestResults, strcat(save_dm_bias,'\bias_ttest_uncertainty.csv'));
%% T-TEST ON ABSOLUTE BETA COEFFICIENTS

% ttest
cond_names = ["pe";"pe_condiff";"pe_salience";"pe_congruence";"pe_rewunc";"pe_estunc";"pe_pesign"];
[h_vals, p_vals, ~, stats] = ttest(betas_abs.betas_all); % Perform a one-sample t-test
t_vals = stats.tstat;
df_vals = stats.df;

% mean and SEM
mean_ecoperf = NaN(size(betas_abs.betas_all,2),1);
sem_ecoperf = NaN(size(betas_abs.betas_all,2),1);
for i = 1:size(betas_abs.betas_all,2)
    [mean_ecoperf(i,:),sem_ecoperf(i,:)] = compute_mean_sem(betas_abs.betas_all(:,i));
end

% save output to .csv file for OVERLEAF
ttestResults = table(cond_names, round(h_vals,2).', round(p_vals,4).', round(t_vals,2).', ...
    round(df_vals,2).', round(mean_ecoperf,2), round(sem_ecoperf,3),...
    'VariableNames', {'Condition', 'HValue', 'PValue', 'TStat', 'df','mean','sem'}); 
writetable(ttestResults, strcat(save_lr_suppl,'\abslr_betas_ttest.csv'));