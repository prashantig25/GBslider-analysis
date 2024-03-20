%% Note
% make sure the behaviour data file has state,action,choice,contrast left,
% contrast right, contrast difference, condition, reward probability,
% contrast of the block.

% choice is the correct economic performance in the pilot study (study 1) files

% FOR study 2
% create condition column with 1,2,3
% create contrast column 0,1 

%% READ and MERGE all data files for pilot study
clc
clearvars 

% INITIALISE ALL REQUIRED VARS
colors; % all colors required for plotting figures
t = 25; % number of trials in a block
num_cond = 4; % number of conditions in the task
num_cont = 2; % number of contrast levels in the task
num_blocks = 4; % number of blocks per condition
total_blocks = 16; % total number of blocks per subject

% CHANGE DATA DIRECTORY ACCORDINGLY
behv_dir = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\MAT files\raw\BIDS\pilot_study"; 
save_dir = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\MAT files\descriptive\study1";

% MERGE ALL SUBJECT DATA
subj_ids = [11:17,19:22,24,26:33,36,37,42,46,48,51,53,54,56,58,59,61,62,...
    65,68,69,74,76,83,85,86,87,89,95,96,99,100,109,110,113,114,115,...
    117	119	121	122	124	126	127	128	129	136	138	141	143	144	145	146	147,...
    148	149	150	151	153	154	156	157	159	161	162	163	164	165	167	171	172,...	
    176	177	180	182	183	188	191];
num_subjs = length(subj_ids);
data_all = []; % empty table to merge all subjects data
for i = 1:length(subj_ids)
    tsv_file = fullfile(behv_dir,strcat('sub_',num2str(subj_ids(i))),'behav', ...
        strcat('sub_',num2str(subj_ids(i)),".tsv")); % path and file name for TSV file
    data_table = readtable(tsv_file, "FileType","text",'Delimiter', '\t'); % read
    data_all = [data_all; data_table]; % merge all subjects data
end
%% calculate economic performance

% ACROSS CONDITIONS
ecoperf_cond = NaN(num_subjs,num_cond);

% MEAN ECONOMIC PERFORMANCE FOR EACH CONDITION
for i = 1:num_subjs
    for c = 1:num_cond
        ecoperf_cond(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.condition_int == c)));
    end
end

% ACROSS CONTRAST
ecoperf_cont = NaN(num_subjs,num_cont);

% MEAN ECONOMIC PERFORMANCE FOR EACH CONTRAST
for i = 1:num_subjs
    for c = 1:num_cont
        ecoperf_cont(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.contrast == c-1)));
    end
end

% ACROSS CONDITION AND CONTRAST
ecoperf_hh = NaN(num_subjs,num_cont);
ecoperf_hl = NaN(num_subjs,num_cont);
ecoperf_lh = NaN(num_subjs,num_cont);
ecoperf_ll = NaN(num_subjs,num_cont);

% MEANS FOR EACH CONDITION, ACROSS CONTRASTS
for i = 1:num_subjs
    data_subj = data_all(data_all.ID == subj_ids(i),:);
    for c = 1:num_cont
        ecoperf_hh(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 1)));
        ecoperf_hl(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 2)));
        ecoperf_lh(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 3)));
        ecoperf_ll(i,c) = nanmean(data_subj.ecoperf(and(data_subj.contrast == c-1,data_subj.condition_int == 4)));
    end
end
ecoperf_cond_cont = [ecoperf_hh,ecoperf_hl,ecoperf_lh,ecoperf_ll];

%% plot figures

y = [ecoperf_hh;ecoperf_hl;ecoperf_lh;ecoperf_ll]; % across conditions
num_subjs = length(ecoperf_ll); % number of subjects 
hh_avg = nanmean(ecoperf_hh,1); % mean across subjects 
hl_avg = nanmean(ecoperf_hl,1);
lh_avg = nanmean(ecoperf_lh,1);
ll_avg = nanmean(ecoperf_ll,1);

mean_avg = [hh_avg; hl_avg;lh_avg;ll_avg;]; % mean across subjects for each condition

hh_sd = std(ecoperf_hh,1)/sqrt(num_subjs); % SEM across subjects 
hl_sd = std(ecoperf_hl,1)/sqrt(num_subjs);
lh_sd = std(ecoperf_lh,1)/sqrt(num_subjs);
ll_sd = std(ecoperf_ll,1)/sqrt(num_subjs);

mean_sd = [hh_sd; hl_sd;lh_sd;ll_sd;]; % SEM across subjects for each condition
xticks = [1:length(mean_sd)]; % number of xticks for the plot
xticklabs = {'Both','Perceptual','Reward','No'}; % xtick labels
title_name = {'Study 1'}; % plot title
legend_names = {'Low Contrast','High Contrast'}; % legends for bar
xlabelname = {'Condition'}; % xlabel title
ylabelname = {'Mean economic performance'}; % ylabel title
colors_name = [c4_4;c3_4]; % colors for bars,

% CREATE FIGURE
figure
bar_plots(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,6,1,'Arial',colors_name)  

% CALCULATE SALIENCE BIAS 
salience_bias_hh = ecoperf_hh(:,2)-ecoperf_hh(:,1);
salience_bias_hl = ecoperf_hl(:,2)-ecoperf_hl(:,1);
salience_bias_lh = ecoperf_lh(:,2)-ecoperf_lh(:,1);
salience_bias_ll = ecoperf_ll(:,2)-ecoperf_ll(:,1);

% MEAN SALIENCE BIAS
y = [salience_bias_hh;salience_bias_hl;salience_bias_lh;salience_bias_ll];
num_subjs = length(salience_bias_ll);

% MEAN ACROSS SUBJECTS
hh_avg = nanmean(salience_bias_hh,1);
hl_avg = nanmean(salience_bias_hl,1);
lh_avg = nanmean(salience_bias_lh,1);
ll_avg = nanmean(salience_bias_ll,1);
mean_avg = [hh_avg; hl_avg;lh_avg;ll_avg;];

% SEM ACROSS SUBJECTS
hh_sd = std(salience_bias_hh,1)/sqrt(num_subjs);
hl_sd = std(salience_bias_hl,1)/sqrt(num_subjs);
lh_sd = std(salience_bias_lh,1)/sqrt(num_subjs);
ll_sd = std(salience_bias_ll,1)/sqrt(num_subjs);
mean_sd = [hh_sd; hl_sd;lh_sd;ll_sd;];

xticks = [1:length(mean_sd)]; % x-axis ticks
xticklabs = {'Both','Perceptual','Reward','No'}; % x-axis tick labels
title_name = {'Study 1'}; % figure title
legend_names = {'Salience bias'}; % label name
xlabelname = {'Condition'}; % x-axis label name
ylabelname = {'Mean salience bias'}; % y-axis label name
colors_name = [neutral]; % colors for bar

% CREATE FIGURE
figure
bar_plots(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,6,1,'Arial',colors_name)  

% INITIALISE ARRAYS FOR SINGLE TRIAL ECONOMIC PERFORMANCE
hh_curve = NaN(num_subjs,t);
hl_curve = NaN(num_subjs,t);
lh_curve = NaN(num_subjs,t);
ll_curve = NaN(num_subjs,t);

for i = [1:9,11:93] % exclude participant 21 because unbalanced blocks
    data_subj = data_all(data_all.ID==subj_ids(i),:);
    uni_mix = unique(data_subj.blocks(data_subj.condition_int==1));
    uni_perc = unique(data_subj.blocks(data_subj.condition_int==2));
    uni_rew = unique(data_subj.blocks(data_subj.condition_int==3));
    uni_no = unique(data_subj.blocks(data_subj.condition_int==4));
    hh_subj = NaN(num_cond,t);
    hl_subj = NaN(num_cond,t);
    lh_subj = NaN(num_cond,t);
    ll_subj = NaN(num_cond,t);
    for b = 1:num_cond
        hh_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_mix(b),data_subj.condition_int == 1));
        hl_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_perc(b),data_subj.condition_int == 2));
        lh_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_rew(b),data_subj.condition_int == 3));
        ll_subj(b,:) = data_subj.ecoperf(and(data_subj.blocks == uni_no(b),data_subj.condition_int == 4));
    end
    hh_curve(i,:) = mean(hh_subj);
    hl_curve(i,:) = mean(hl_subj);
    lh_curve(i,:) = mean(lh_subj);
    ll_curve(i,:) = mean(ll_subj);
end

% MEAN ACROSS SUBJECTS
hh_mean = nanmean(hh_curve);
hl_mean = nanmean(hl_curve);
lh_mean = nanmean(lh_curve);
ll_mean = nanmean(ll_curve);

% SEM ACROSS SUBJECTS
hh_sem = nanstd(hh_curve)./sqrt(num_subjs);
hl_sem = nanstd(hl_curve)./sqrt(num_subjs);
lh_sem = nanstd(lh_curve)./sqrt(num_subjs);
ll_sem = nanstd(ll_curve)./sqrt(num_subjs);

mean_curves = [hh_mean;hl_mean;lh_mean;ll_mean];
sem_curves = [hh_sem;hl_sem;lh_sem;ll_sem];
colors_name = [mix;perc;rew;no]; % colors for line plots
legend_names = {'Both','Perceptual','Reward','No'}; % legend for line plots
title_name = {'Learning curve'}; % figure title
xlabelname = {'Trial'}; % name for x-axis label
ylabelname = {'Mean economic performance'}; % name for y-axis title
x = 1:t; % number of trials in a block

% PLOT FIGURE
figure
hold on
lg_curves(x,mean_curves,sem_curves,colors_name,legend_names,title_name,xlabelname,ylabelname)

%% SAVE DATA

writetable(data_all,strcat(save_dir,"\study1.xlsx"));

save(strcat(save_dir,'\ecoperf_hh')); 
save(strcat(save_dir,'\ecoperf_hl'));
save(strcat(save_dir,'\ecoperf_lh'));
save(strcat(save_dir,'\ecoperf_ll'));

save(strcat(save_dir,'\hh_curve')); 
save(strcat(save_dir,'\hl_curve'));
save(strcat(save_dir,'\lh_curve'));
save(strcat(save_dir,'\ll_curve'));