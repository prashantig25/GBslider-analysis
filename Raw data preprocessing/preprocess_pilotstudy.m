%% Note
% make sure the behaviour data file has state,action,choice,contrast left,
% contrast right, contrast difference, condition, reward probability,
% contrast of the block.

% choice is the correct economic performance in the pilot study (study 1) files

% FOR study 2
% create condition column with 1,2,3
% create contrast column 0,1 

%% READ, MERGE and PREPROCESS all data files for pilot study
clc
clearvars 

% INITIALISE ALL REQUIRED VARS
t = 25; % number of trials in a block
num_cond = 4; % number of conditions in the task
num_cont = 2; % number of contrast levels in the task
num_blocks = 4; % number of blocks per condition
total_blocks = 16; % total number of blocks per subject

% CHANGE DATA DIRECTORY ACCORDINGLY
behv_dir = "DATA\pilot_study"; 
save_dir = "saved_files\study1";
mkdir(save_dir);

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
%% CALCULATE ECONOMIC PERFORMANCE

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
%% SAVE DATA

writetable(data_all,fullfile(save_dir,"study1.txt"));

save(fullfile(save_dir,'ecoperf_hh.mat')); 
save(fullfile(save_dir,'ecoperf_hl,mat'));
save(fullfile(save_dir,'ecoperf_lh.mat'));
save(fullfile(save_dir,'ecoperf_ll.mat'));

save(fullfile(save_dir,'hh_curve.mat')); 
save(fullfile(save_dir,'hl_curve.mat'));
save(fullfile(save_dir,'lh_curve.mat'));
save(fullfile(save_dir,'ll_curve.mat'));