%% READ and MERGE all data files for study 2
clc
clearvars 

% INITIALISE REQUIRED VARIABLES
colors; % all colors required for plotting figures
t = 25; % number of trials in a block
num_cond = 3; % number of conditions in the task
num_cont = 2; % number of contrast levels in the task
num_blocks = 4; % number of blocks per condition
total_blocks = 12; % total number of blocks per subject

% CHANGE DIRECTORY ACCORDINGLY
behv_dir = "C:\Users\prash\Nextcloud\Thesis_laptop\clean_scripts\Behaviour\Data\Study2"; 
save_dir = "C:\Users\prash\Nextcloud\Thesis_laptop\Semester 7\behv_manuscript\MAT files\descriptive\study2";

% ARRAY WITH SUBJECT IDS
subj_ids = [139	143	145	146	151	157	159	160	162	163	164	165	174	176	181	192	198,...	
    199	207	208	209	210	211	213	214	216	219	220	229	242	243	247	249	254	255,...	
    256	258	260	266	267	270	275	276	277	278	279	280	303	306	307	310	311	313	316,...
    319	320	322	323	326	327	328	330	331	334	335	338	339	341	349	353	355	358	359,...	
    360	361	363	364	365	366	367	370	372	374	376	377	387	388	404	405	406	407	408,...	
    412	416	418	420	421	423];

% MERGE ALL SUBJECTS 
num_subjs = length(subj_ids);
data_all = []; % empty table to merge all subjects data
for i = 1:length(subj_ids)
    data_subj_choice = readtable(strcat(behv_dir,'\',num2str(subj_ids(i)),".xlsx"),"Sheet","choice"); % read each subject's file for choice data
    data_subj_mu = readtable(strcat(behv_dir,'\',num2str(subj_ids(i)),".xlsx"),"Sheet","mu"); % read each subject's file for reported mu
    data_choice = table(data_subj_choice.run_id,data_subj_choice.state,data_subj_choice.rt,data_subj_choice.response, ... % extract only important data
        data_subj_choice.condition,data_subj_choice.contrast_left,data_subj_choice.contrast_right,...
        data_subj_choice.corr_resp,data_subj_choice.correct,'VariableNames',...
        {'id','state','rt','response','condition','contrast_left','contrast_right' ...
        ,'correct response','correct'});
    data_mu = table(data_subj_mu.rt,data_subj_mu.response, ... % extract only important data
        data_subj_mu.condition,'VariableNames',{'rt_mu','response_mu','condition_mu'});
    data_choice.mu = data_mu.response_mu;
    data_choice.condition_mu = data_mu.condition_mu;

    data_all = [data_all;data_choice];
    if height(data_choice) > 300
        disp(height(data_choice))
    end
end

%% clean the data by multiple keypresses and rts

% EXTRACT LAST KEYPRESS FROM MULTIPLE KEYPRESSES
for i =1:height(data_all)
    multi_keypress = string(data_all.response(i));
    multi_keypress = strsplit(multi_keypress,',');
    if length(multi_keypress) > 1
        multi_keypress_char = convertStringsToChars(multi_keypress(end));
        multi_keypress_char(isletter(multi_keypress_char)==0)=[];
        data_all.response_str(i) = convertCharsToStrings(multi_keypress_char);
    else
        multi_keypress_char = convertStringsToChars(multi_keypress);
        multi_keypress_char(isletter(multi_keypress_char)==0)=[];
        data_all.response_str(i) = convertCharsToStrings(multi_keypress_char);
    end
end

% CONVERT RESPONSE TO ACTION TO FIT FRAMEWORK OF a = 0, a = 1
for i =1:height(data_all)
    if strcmp(data_all.response_str(i),'arrowleft') == 1
        data_all.choice(i) = 0;
    elseif strcmp(data_all.response_str(i),'arrowright') == 1
        data_all.choice(i) = 1;
    end
end
%% add trial and block numbers

% ADD TRIAL NUMBERS
trials = 1:t; % array with trial numbers
trials = repmat(trials.',total_blocks,1); % repeat for all blocks
trials_all = repmat(trials,num_subjs,1); % repeat for all subjects
data_all.trials = trials_all; % add data to the table

% ADD BLOCK NUMBERS
blocks = zeros(total_blocks*t,1); % initialise array for block number 
c = 1;
for b = 1:total_blocks
    blocks(c:b*t,1) = repelem(b,t,1); % repeat block numbers for all trials
    c = c+t;
end
blocks_all = repmat(blocks,num_subjs,1); % repeat for each subject
data_all.blocks = blocks_all; % add to table

%% get conditions and contrast levels

% CHECK IF TRIAL BELONGS TO LOW/HIGH CONTRAST BLOCK
for i = 1:height(data_all)
    if strcmp(data_all.condition(i),'mixed_lc') == 1 || strcmp(data_all.condition(i),'perceptual_lc') == 1 || strcmp(data_all.condition(i),'reward_lc') == 1
        data_all.contrast(i) = 0;
    elseif strcmp(data_all.condition(i),'mixed_hc') == 1 || strcmp(data_all.condition(i),'perceptual_hc') == 1 || strcmp(data_all.condition(i),'reward_hc') == 1
        data_all.contrast(i) = 1;
    end
end

% ADD CONDITION NAME FOR ALL TRIALS IN A BLOCK
for i = 1:height(data_all)
    if data_all.trials(i) ~= 1
        data_all.condition(i) = data_all.condition(i-1);
    else
        data_all.condition(i) = data_all.condition(i+1);
    end
end

% REPLACE CONDITION NAMES
for i = 1:height(data_all)
    if strcmp(data_all.condition(i),'mixed_lc') == 1 || strcmp(data_all.condition(i),'mixed_hc') == 1
        data_all.condition_int(i) = 1;
    elseif strcmp(data_all.condition(i),'perceptual_lc') == 1 || strcmp(data_all.condition(i),'perceptual_hc') == 1 
        data_all.condition_int(i) = 2;
    elseif strcmp(data_all.condition(i),'reward_lc') == 1 || strcmp(data_all.condition(i),'reward_hc') == 1 
        data_all.condition_int(i) = 3;
    end
end

% CHECK IF TRIAL IS CONGRUENT OR INCONGRUENT
for i = 1:height(data_all)
    if strcmp(data_all.condition_mu(i),'mu_congruent') == 1 
        data_all.congruence(i) = 1;
    elseif strcmp(data_all.condition_mu(i),'mu_incongruent') == 1
        data_all.congruence(i) = 0;
    end
end
%% check if participant chose the more rewarding option in a trial

% ECONOMIC PERFORMANCE ON A TRIAL
for i = 1:height(data_all)
    if data_all.contrast(i) == 0
        if data_all.state(i) == data_all.choice(i)
            data_all.ecoperf(i) = 1;
        else
            data_all.ecoperf(i) = 0;
        end
    else
        if data_all.state(i) == data_all.choice(i)
            data_all.ecoperf(i) = 0;
        else
            data_all.ecoperf(i) = 1;
        end
    end
end
%% calculate economic performance

% ACROSS CONDITIONS
ecoperf_cond = NaN(num_subjs,num_cond);

% MEAN ECONOMIC PERFORMANCE ACROSS SUBJECTS, FOR EACH CONDITION
for i = 1:num_subjs
    for c = 1:num_cond
        ecoperf_cond(i,c) = mean(data_all.ecoperf(and(data_all.id == subj_ids(i),data_all.condition_int == c)));
    end
end

% ACROSS LOW/HIGH CONTRAST BLOCKS
ecoperf_cont = NaN(num_subjs,num_cont);

% MEAN ECONOMIC PERFORMANCE ACROSS SUBJECTS, FOR EACH LOW/HIGH CONTRAST
for i = 1:num_subjs
    for c = 1:num_cont
        ecoperf_cont(i,c) = mean(data_all.ecoperf(and(data_all.id == subj_ids(i),data_all.contrast == c-1)));
    end
end

% ACROSS CONDITIONS AND CONTRASTS
ecoperf_mix = NaN(num_subjs,num_cont);
ecoperf_perc = NaN(num_subjs,num_cont);
ecoperf_rew = NaN(num_subjs,num_cont);

for i = 1:num_subjs
    data_subj_choice = data_all(data_all.id == subj_ids(i),:);
    for c = 1:num_cont
        ecoperf_mix(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.condition_int == 1)));
        ecoperf_perc(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.condition_int == 2)));
        ecoperf_rew(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.condition_int == 3)));
    end
end

ecoperf_cond_cont = [ecoperf_mix,ecoperf_perc,ecoperf_rew];

% RUN REQUIRED SIGNIFICANCE TESTS
[p,tbl,stats] = anova1(ecoperf_cond);
[h,p,ci,stats] = ttest(ecoperf_cond(:,1),ecoperf_cond(:,3));
%% get subjective estimate of contingency parameter

% COMPUTE SUBJECTIVE CONTINGENCY PARAMETER FOR INCONGRUENT BLOCKS
for i = 1:height(data_all)
    if data_all.congruence(i) == 1
        data_all.mu_congruence(i) = data_all.mu(i);
    else
        data_all.mu_congruence(i) = 100-data_all.mu(i);
    end
end
%% plot figures

% MEAN ECONOMIC PERFORMANCE
y = [ecoperf_mix;ecoperf_perc;ecoperf_rew;];
num_subjs = length(ecoperf_mix);
mix_avg = nanmean(ecoperf_mix,1);
perc_avg = nanmean(ecoperf_perc,1);
rew_avg = nanmean(ecoperf_rew,1);

mean_avg = [mix_avg; perc_avg;rew_avg;];

% SEM ACROSS SUBJECTS
mix_sd = nanstd(ecoperf_mix,1)/sqrt(num_subjs);
perc_sd = nanstd(ecoperf_perc,1)/sqrt(num_subjs);
rew_sd = nanstd(ecoperf_rew,1)/sqrt(num_subjs);

mean_sd = [mix_sd; perc_sd;rew_sd;];
xticks = [1:length(mean_sd)];

% FIGURE PROPERTIES
xticklabs = {'Both','Perceptual','Reward'};% x-axis tick labels
title_name = {'Study 2'}; % figure title
legend_names = {'Low Contrast','High Contrast'}; % legend names
xlabelname = {'Type of uncertainty'}; % x-axis label name
ylabelname = {'Economic performance'}; % y-axis label name
colors_name = [c4_4;c3_4]; % bar colors

% CREATE FIGURE
% figure
% bar_plots(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
%     legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,colors_name)  

% SALIENCE BIAS
salience_bias_mix = ecoperf_mix(:,2)-ecoperf_mix(:,1);
salience_bias_perc = ecoperf_perc(:,2)-ecoperf_perc(:,1);
salience_bias_rew = ecoperf_rew(:,2)-ecoperf_rew(:,1);

% MEAN ECONOMIC PERFORMANCE
y = [salience_bias_mix;salience_bias_perc;salience_bias_rew;];
mean_all = [];
mix_avg = nanmean(salience_bias_mix,1);
perc_avg = nanmean(salience_bias_perc,1);
rew_avg = nanmean(salience_bias_rew,1);

mean_avg = [mix_avg; perc_avg;rew_avg];

% SEM ACROSS CONDITIONS
mix_sd = nanstd(salience_bias_mix,1)/sqrt(num_subjs);
perc_sd = nanstd(salience_bias_perc,1)/sqrt(num_subjs);
rew_sd = nanstd(salience_bias_rew,1)/sqrt(num_subjs);

mean_sd = [mix_sd; perc_sd;rew_sd;];
xticks = [1:length(mean_sd)];

% FIGURE PROPERTIES
xticklabs = {'Both','Perceptual','Reward'}; % x-axis tick labels
title_name = {'Study 2'}; % figure title
legend_names = {'Salience bias'}; % legend names
xlabelname = {'Type of uncertainty'}; % x-axis label name
ylabelname = {'Mean salience bias'}; % y-axis label name
colors_name = [neutral]; % bar color

% CREATE FIGURE
figure
bar_plots(y,mean_avg,mean_sd,num_subjs,length(mean_avg),length(legend_names), ...
    legend_names,xticks,xticklabs,title_name,xlabelname,ylabelname,6,1,'Arial',colors_name)  

% LEARNING CURVES

% INITIALISE TO STORE MEAN MU FOR ALL TRIALS, ACROSS SUBJECTS
mix_curve = NaN(num_subjs,t);
perc_curve = NaN(num_subjs,t);
rew_curve = NaN(num_subjs,t);

for i = [1:num_subjs] % exclude participant 21 because unbalanced blocks
    data_subj_choice = data_all(data_all.id==subj_ids(i),:); % for each subject
    uni_mix = unique(data_subj_choice.blocks(data_subj_choice.condition_int==1)); % block number for condition = 1
    uni_perc = unique(data_subj_choice.blocks(data_subj_choice.condition_int==2)); % block number for condition = 2
    uni_rew = unique(data_subj_choice.blocks(data_subj_choice.condition_int==3)); % block number for condition = 3
    mix_subj = NaN(num_blocks,t);
    perc_subj = NaN(num_blocks,t);
    rew_subj = NaN(num_blocks,t);
    for b = 1:num_blocks
        mix_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_mix(b),data_subj_choice.condition_int == 1));
        perc_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_perc(b),data_subj_choice.condition_int == 2));
        rew_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_rew(b),data_subj_choice.condition_int == 3));
    end
    mix_curve(i,:) = mean(mix_subj);
    perc_curve(i,:) = mean(perc_subj);
    rew_curve(i,:) = mean(rew_subj);
end

x = 1:t; % number of trials in a block

% MEAN MU ACROSS SUBJECTS
mix_mean = nanmean(mix_curve);
perc_mean = nanmean(perc_curve);
rew_mean = nanmean(rew_curve);

mean_curves = [mix_mean;perc_mean;rew_mean;];

% SEM ACORSS SUBJECTS
mix_sem = nanstd(mix_curve)./sqrt(num_subjs);
perc_sem = nanstd(perc_curve)./sqrt(num_subjs);
rew_sem = nanstd(rew_curve)./sqrt(num_subjs);

sem_curves = [mix_sem;perc_sem;rew_sem;];

% FIGURE PROPERTIES
colors_name = [mix;perc;rew]; % colors for plot lines
legend_names = {'Both','Perceptual','Reward'}; % legend names
title_name = {'Learning curve'}; % figure title
xlabelname = {'Trial'}; % x-axis label name
ylabelname = {'Mean economic performance'}; % y-axis label name

% CREATE FIGURE
figure
lg_curves(x,mean_curves,sem_curves,colors_name,legend_names,title_name,xlabelname,ylabelname,6,1,'Arial')


% MEAN MU ACROSS SUBJECTS
mix_mean = nanmean(mix_curve,2);
perc_mean = nanmean(perc_curve,2);
rew_mean = nanmean(rew_curve,2);

mean_curves = [mix_mean,perc_mean,rew_mean;];

% RUN REQUIRED SIGNIFICANCE TESTS
[p,tbl,stats] = anova1(mean_curves);
[h,p,ci,stats] = ttest(mix_mean,rew_mean);

%% SAVE DATA

writetable(data_all,strcat(save_dir,'\study2.xlsx'));
save(strcat(save_dir,'\ecoperf_mix'));
save(strcat(save_dir,'\ecoperf_perc'));
save(strcat(save_dir,'\ecoperf_rew'));

save(strcat(save_dir,'\mix_curve'));
save(strcat(save_dir,'\perc_curve'));
save(strcat(save_dir,'\rew_curve'));