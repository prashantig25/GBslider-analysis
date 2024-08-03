%% READ, MERGE and PREPROCESS all data files for main study 
clc
clearvars 

% INITIALISE REQUIRED VARIABLES
t = 25; % number of trials in a block
num_cond = 3; % number of conditions in the task
num_cont = 2; % number of contrast levels in the task
num_blocks = 4; % number of blocks per condition
total_blocks = 12; % total number of blocks per subject

% Get the current working directory
currentDir = pwd;

% CHANGE DIRECTORY ACCORDINGLY
behv_dir = strcat('DATA', filesep, 'main_study'); 
save_dir = strcat('Data', filesep, 'main study'); 
mkdir(save_dir);

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
    tsv_file = fullfile(currentDir, behv_dir,strcat('sub_',num2str(subj_ids(i))),'behav', ...
            strcat('sub_',num2str(subj_ids(i)),".tsv")); % path and file name for TSV file
    data_table = readtable(tsv_file, "FileType","text",'Delimiter', '\t'); % read
    data_all = [data_all; data_table]; % merge all subjects data
end

%% CALCULATE ECONOMIC PERFORMANCE

% ACROSS CONDITIONS
ecoperf_cond = NaN(num_subjs,num_cond);

% MEAN ECONOMIC PERFORMANCE ACROSS SUBJECTS, FOR EACH CONDITION
for i = 1:num_subjs
    for c = 1:num_cond
        ecoperf_cond(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.choice_cond == c)));
    end
end

% ACROSS LOW/HIGH CONTRAST BLOCKS
ecoperf_cont = NaN(num_subjs,num_cont);

% MEAN ECONOMIC PERFORMANCE ACROSS SUBJECTS, FOR EACH LOW/HIGH CONTRAST
for i = 1:num_subjs
    for c = 1:num_cont
        ecoperf_cont(i,c) = mean(data_all.ecoperf(and(data_all.ID == subj_ids(i),data_all.contrast == c-1)));
    end
end

% ACROSS CONDITIONS AND CONTRASTS
ecoperf_mix = NaN(num_subjs,num_cont);
ecoperf_perc = NaN(num_subjs,num_cont);
ecoperf_rew = NaN(num_subjs,num_cont);

for i = 1:num_subjs
    data_subj_choice = data_all(data_all.ID == subj_ids(i),:);
    for c = 1:num_cont
        ecoperf_mix(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.choice_cond == 1)));
        ecoperf_perc(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.choice_cond == 2)));
        ecoperf_rew(i,c) = mean(data_subj_choice.ecoperf(and(data_subj_choice.contrast == c-1,data_subj_choice.choice_cond == 3)));
    end
end

ecoperf_cond_cont = [ecoperf_mix,ecoperf_perc,ecoperf_rew];
%% GET SLIDER RESPONSES

% COMPUTE SUBJECTIVE CONTINGENCY PARAMETER FOR INCONGRUENT BLOCKS
for i = 1:height(data_all)
    if data_all.congruence(i) == 1
        data_all.mu_congruence(i) = data_all.mu(i);
    else
        data_all.mu_congruence(i) = 100-data_all.mu(i);
    end
end
data_all.mu_congruence = data_all.mu_congruence./100;
data_all.mu = data_all.mu./100;

% INITIALISE TO STORE MEAN MU FOR ALL TRIALS, ACROSS SUBJECTS
mix_curve = NaN(num_subjs,t);
perc_curve = NaN(num_subjs,t);
rew_curve = NaN(num_subjs,t);

for i = [1:num_subjs] % exclude participant 21 because unbalanced blocks
    data_subj_choice = data_all(data_all.ID==subj_ids(i),:); % for each subject
    uni_mix = unique(data_subj_choice.blocks(data_subj_choice.choice_cond==1)); % block number for condition = 1
    uni_perc = unique(data_subj_choice.blocks(data_subj_choice.choice_cond==2)); % block number for condition = 2
    uni_rew = unique(data_subj_choice.blocks(data_subj_choice.choice_cond==3)); % block number for condition = 3
    mix_subj = NaN(num_blocks,t);
    perc_subj = NaN(num_blocks,t);
    rew_subj = NaN(num_blocks,t);
    for b = 1:num_blocks
        mix_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_mix(b),data_subj_choice.choice_cond == 1));
        perc_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_perc(b),data_subj_choice.choice_cond == 2));
        rew_subj(b,:) = data_subj_choice.mu_congruence(and(data_subj_choice.blocks == uni_rew(b),data_subj_choice.choice_cond == 3));
    end
    mix_curve(i,:) = mean(mix_subj);
    perc_curve(i,:) = mean(perc_subj);
    rew_curve(i,:) = mean(rew_subj);
end

%% SAVE DATA

safe_saveall(fullfile(save_dir,'study2.txt'),data_all);

safe_saveall(fullfile(save_dir,'ecoperf_mix.mat'),ecoperf_mix);
safe_saveall(fullfile(save_dir,'ecoperf_perc.mat'),ecoperf_perc);
safe_saveall(fullfile(save_dir,'ecoperf_rew.mat'),ecoperf_rew);

safe_saveall(fullfile(save_dir,'mix_curve.mat'),mix_curve);
safe_saveall(fullfile(save_dir,'perc_curve.mat'),perc_curve);
safe_saveall(fullfile(save_dir,'rew_curve.mat'),rew_curve);