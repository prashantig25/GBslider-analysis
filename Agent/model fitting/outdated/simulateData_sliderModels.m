clc
clearvars

% import parameters
params_basicRL = importdata("params_basicRL_sigma_integratedPerceptual.mat");
params_RLsigma = importdata("params_RLsigma_integratedPerceptual.mat");
params_RLsigma_confirmBias = importdata("params_RLsigma_confirmBias_integratedPerceptual.mat");
params_bayesianAgent = importdata("params_bayesianAgent_integratedPerceptual.mat");
params_bayesianAgent_confirmBias = importdata("params_bayesianAgent_confirmBias_integratedPerceptual.mat");

% params_basicRL = importdata('params_basicRL_NOintegral.mat');      % fields: alpha, kappa
% params_RLsigma = importdata('params_RLsigma_NOintegral.mat');      % fields: alpha, kappa, sigma
% params_RLsigma_confirmBias = importdata("params_RLsigma_confirmBias_NOintegral.mat");            % fields: alpha, kappa, sigma
% params_bayesianAgent = importdata('params_bayesianAgent_NOintegral_evalana.mat'); % fields: kappa, sigma
% params_bayesianAgent_confirmBias = importdata('params_bayesianAgent_confirmBias_NOintegral_evalana.mat'); % fields: kappa, sigma

% import task-related data
data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
data = data(data.choice_cond ~= 3,:);
% data = data(data.choice_cond ~= 2,:);

% data = data(data.pe ~= 0, :);
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);

% Precompute contrast difference
data.condiff_relative = (data.contrast_left - data.contrast_right) ./ 2;

model_names = {'basicRL','RLEstSens','RLEstSens_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
% model_names = {'basicRL','RLEstSens'}; %,'RLEstSens_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
%model_names = {'BayesianAgent'};
nModels = numel(model_names);

% Clean model names for struct field usage
model_fieldnames = regexprep(model_names, '[^a-zA-Z0-9]', '_');

% Use cleaned field names for struct
simulated_data = struct();
mu_data = struct();
for m = 1:nModels
    simulated_data.(model_fieldnames{m}) = cell(numSubjs, 1);
    mu_data.(model_fieldnames{m}) = cell(numSubjs, 1);
end
nSimulations = 2;  % Number of simulations per model/subject

%% =================== RUN SIMULATIONS =================================
dataSims = [];
for n = 1:numSubjs

    dataSubj = data(data.ID == uniqueID(n),:);
    % --- Subject-specific setup ---
    subj = preprocess_fitSlider(data, uniqueID(n));
    rewards = arrayfun(@(h) ...
        subj.recoded_rewards(h) * (subj.contrast(h) == 0) + ...
        (1 - subj.recoded_rewards(h)) * (subj.contrast(h) ~= 0), ...
        (1:length(subj.mu_hat)));
    condition = subj.dataTable.condition;

    % --- Run simulations for each model ---
    for m = 1:nModels
        model_name = model_names{m};
        nTrials = length(subj.blocks);
        simulations = NaN(nTrials, nSimulations);
        mu = NaN(nTrials, nSimulations);

        % Get appropriate parameters
        switch model_name
            case 'basicRL'
                params = [params_basicRL.alpha(n), params_basicRL.kappa(n)]; %, params_basicRL.beta(n)];
            case 'RLEstSens'
                params = [params_RLsigma.alpha(n), params_RLsigma.kappa(n), params_RLsigma.sigma(n)]; %, params_RLsigma.beta(n)];
            case 'RLEstSens_confirmBias'
                params = [params_RLsigma_confirmBias.kappa(n), params_RLsigma_confirmBias.sigma(n), params_RLsigma_confirmBias.confirmBias(n), params_RLsigma_confirmBias.noconfirmBias(n)]; %, params_RLsigma_confirmBias.beta(n)];
            case 'BayesianAgent_confirmBias'
                params = [params_bayesianAgent_confirmBias.kappa(n), params_bayesianAgent_confirmBias.sigma(n), params_bayesianAgent_confirmBias.confirmBias(n)]; %, params_bayesianAgent_confirmBias.beta(n)];
            case 'BayesianAgent'
                params = [params_bayesianAgent.kappa(n), params_bayesianAgent.sigma(n)]; %, params_bayesianAgent.beta(n)];
        end

        % Run simulations
        parfor s = 1:nSimulations
            switch model_name
                case 'basicRL'
                    [predictedUp, mu_hat] = predict_allModels.predict_basicRL(...
                        params, subj.blocks, rewards, subj.state, 'sample');
                case 'RLEstSens'
                    [predictedUp, mu_hat] = predict_allModels.predict_RLsigma(...
                        params, subj.blocks, rewards, subj.state, subj.condiff, 'sample');
                case 'RLEstSens_confirmBias'
                    [predictedUp, mu_hat] = predict_allModels.predict_RLsigma_confirmBias(...
                        params, subj.blocks, rewards, subj.state, subj.condiff, subj.dataTable.confirm_rew,'sample');
                case 'BayesianAgent'
                    [predictedUp, mu_hat] = predict_allModels.predict_bayesianAgent(...
                        params, subj.blocks, subj.rewards, subj.condiff, ...
                        subj.choices, condition, subj.dataTable.contrast, 'sample');
                case 'BayesianAgent_confirmBias'
                    [predictedUp, mu_hat] = predict_allModels.predict_bayesianAgent_confirmBias(...
                        params, subj.blocks, subj.rewards, subj.condiff, ...
                        subj.choices, condition, subj.dataTable.contrast, subj.dataTable.confirm_rew, 'sample');
            end
            simulations(:, s) = predictedUp;
            mu(:, s) = mu_hat;
        end
        simulated_data.(model_name){n} = simulations;
        mu_data.(model_name){n} = mu;
        valid = ~isnan(dataSubj.mu_congruence);
        dataSubj = dataSubj(valid, :);
        dataSubj.(model_name) = mu;
    end

    dataSims = [dataSims; dataSubj];
end


%% ========================= COMPUTE PE FOR ALL THE THREE MODELS ==============

data_RLSigma = [];
data_basicRL= [];
dataRLSigma_confirmBias = [];
data_bayesianAgent = [];
data_bayesianAgent_confirmBias = [];

mix_sims = NaN(nSimulations,25);
perc_sims = NaN(nSimulations,25);
num_blocks = 4;
numModels = 5;
t = 25;
model_names = {'basicRL','RLEstSens','RLEstSens_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};
mix_sims_all = NaN(numSubjs,25);
perc_sims_all = NaN(numSubjs,25);
pe = [];
up = [];
lr = [];


figure
colors = lines(5);
for m = 1:numModels
    for n = 1:numSubjs

        for ns = 1:nSimulations
            dataSubj = data(data.ID == uniqueID(n),:);
            valid = ~isnan(dataSubj.mu_congruence);
            dataSubj = dataSubj(valid, :);
            if ismember(model_names{m},'basicRL')
                mu_data_subj = mu_data.basicRL{n,1};
            elseif ismember(model_names{m},'RLEstSens')
                mu_data_subj = mu_data.RLEstSens{n,1};
            elseif ismember(model_names{m},'RLEstSens_confirmBias')
                mu_data_subj = mu_data.RLEstSens_confirmBias{n,1};
            elseif ismember(model_names{m},'BayesianAgent')
                mu_data_subj = mu_data.BayesianAgent{n,1};
            elseif ismember(model_names{m},'BayesianAgent_confirmBias')
                mu_data_subj = mu_data.BayesianAgent_confirmBias{n,1};
            end
            dataSubj.mu_basicRL = mu_data_subj(:,ns);

            for i = 2:height(dataSubj)
                if dataSubj.contrast(i) == 1 % if actual mu < 0.5
                    dataSubj.mu_t_1_basicRL(i) = 1-dataSubj.mu_basicRL(i-1);
                    dataSubj.mu_t_basicRL(i) = 1-dataSubj.mu_basicRL(i);
                else
                    dataSubj.mu_t_1_basicRL(i) = dataSubj.mu_basicRL(i-1);
                    dataSubj.mu_t_basicRL(i) = dataSubj.mu_basicRL(i);
                end
            end

            state_zero_idx = dataSubj.state == 0; % index for rows where state is 0

            % COMPUTE PE
            dataSubj.pe_basicRL(state_zero_idx) = dataSubj.recoded_reward(state_zero_idx) - dataSubj.mu_t_1_basicRL(state_zero_idx); % state = 0
            dataSubj.pe_basicRL(~state_zero_idx) = (1 - dataSubj.recoded_reward(~state_zero_idx)) - dataSubj.mu_t_1_basicRL(~state_zero_idx); % state = 1

            % COMPUTE UP
            dataSubj.up_basicRL = NaN(height(dataSubj),1);
            dataSubj.up_basicRL(2:end) = dataSubj.mu_t_basicRL(2:height(dataSubj)) - dataSubj.mu_t_1_basicRL(2:height(dataSubj));
            dataSubj.pe_basicRL(dataSubj.trials == 1,1) = 0;

            % COMPUTE LR
            dataSubj.lr_model = dataSubj.up_basicRL./dataSubj.pe_basicRL;

            % COMPUTE EST ERROR
            for h = 1:height(dataSubj)
                if dataSubj.choice_cond(h) == 1
                    dataSubj.estError(h) = abs(0.7 - dataSubj.mu_basicRL(h));
                else
                    dataSubj.estError(h) = abs(0.9 - dataSubj.mu_basicRL(h));
                end
            end

            pe = [pe,dataSubj.pe_basicRL];
            up = [up,dataSubj.up_basicRL];
            lr = [lr,dataSubj.lr_model];

            uni_mix = unique(dataSubj.blocks(dataSubj.choice_cond==1)); % block number for condition = 1
            uni_perc = unique(dataSubj.blocks(dataSubj.choice_cond==2)); % block number for condition = 2
            mix_subj = NaN(num_blocks,t);
            perc_subj = NaN(num_blocks,t);

            ee_mix = NaN(num_blocks,t);
            ee_perc = NaN(num_blocks,t);

            for b = 1:num_blocks
                temp_mix = dataSubj.mu_basicRL(and(dataSubj.blocks == uni_mix(b), dataSubj.choice_cond == 1));
                temp_perc = dataSubj.mu_basicRL(and(dataSubj.blocks == uni_perc(b), dataSubj.choice_cond == 2));
                    
                eeTemp_mix = dataSubj.estError(and(dataSubj.blocks == uni_mix(b), dataSubj.choice_cond == 1));
                eeTemp_perc = dataSubj.estError(and(dataSubj.blocks == uni_perc(b), dataSubj.choice_cond == 2));
                
                % Pad with NaN if fewer than 25 trials, or truncate if more
                if length(temp_mix) < t
                    temp_mix(end+1:t) = NaN;
                    eeTemp_mix(end+1:t) = NaN;
                elseif length(temp_mix) > t
                    temp_mix = temp_mix(1:t);
                    eeTemp_mix = temp_mix(1:t);
                end

                if length(temp_perc) < t
                    temp_perc(end+1:t) = NaN;
                    eeTemp_perc(end+1:t) = NaN;
                elseif length(temp_perc) > t
                    temp_perc = temp_perc(1:t);
                    eeTemp_perc = temp_perc(1:t);
                end


                mix_subj_blocks(b,:) = temp_mix(:)';  % Ensure row vector
                perc_subj_blocks(b,:) = temp_perc(:)'; % Ensure row vector

                eeMix_subj_blocks(b,:) = eeTemp_mix(:)';
                eePerc_subj_blocks(b,:) = eeTemp_perc(:)';

                % hold on 
                % plot(1:25,mix_subj_blocks,'Color',colors(m,:),'LineWidth',2)
                % hold on
                % plot(1:25,perc_subj_blocks,'Color',colors(m,:),'LineWidth',2)
            end
            mix_sims(ns,:) = mean(mix_subj_blocks);
            perc_sims(ns,:) = mean(perc_subj_blocks);

            eeMix_sims(ns,:) = mean(eeMix_subj_blocks);
            eePerc_sims(ns,:) = mean(eePerc_subj_blocks);
        end
        % switch model_name
        %     case 'basicRL'
        %         mix_sims_all.basicRL{n,:} = mean(mix_sims);
        %         perc_sims_all.basicRL{n,:} = mean(perc_sims);
        %     case 'RLEstSens'
        %         mix_sims_all.RLEstSens{n,:} = mean(mix_sims);
        %         perc_sims_all.RLEstSens{n,:} = mean(perc_sims);
        %     case 'RLEstSens_confirmBias'
        %         mix_sims_all.RLEstSens_confirmBias{n,:} = mean(mix_sims);
        %         perc_sims_all.RLEstSens_confirmBias{n,:} = mean(perc_sims);
        % end
        mix_sims_all(n,:) = mean(mix_sims);
        perc_sims_all(n,:) = mean(perc_sims);

        eeMix_sims_all(n,:) = mean(eeMix_sims);
        eePerc_sims_all(n,:) = mean(eePerc_sims);

        peSims = mean(pe,2,"omitmissing");
        upSims = mean(up,2,"omitmissing");
        lrSims = mean(lr,2,"omitmissing");

        pe = [];
        up = [];
        lr = [];

        if ismember(model_names{m},'basicRL')
            dataSubj.pe_basicRL = peSims;
            dataSubj.up_basicRL = upSims;
            dataSubj.lr_basicRL = lrSims;
            data_basicRL = [data_basicRL; dataSubj];
        elseif ismember(model_names{m},'RLEstSens')
            dataSubj.pe_RLSigma = peSims;
            dataSubj.up_RLSigma = upSims;
            dataSubj.lr_RLSigma = lrSims;
            data_RLSigma = [data_RLSigma; dataSubj];
        elseif ismember(model_names{m},'RLEstSens_confirmBias')
            dataSubj.pe_RLSigma_confirmBias = peSims;
            dataSubj.up_RLSigma_confirmBias = upSims;
            dataSubj.lr_RLSigma_confirmBias = lrSims;
            dataRLSigma_confirmBias = [dataRLSigma_confirmBias; dataSubj];
        elseif ismember(model_names{m},'BayesianAgent')
            dataSubj.pe_BayesianAgent = peSims;
            dataSubj.up_BayesianAgent = upSims;
            dataSubj.lr_BayesianAgent = lrSims;
            data_bayesianAgent = [data_bayesianAgent; dataSubj];
        elseif ismember(model_names{m},'BayesianAgent_confirmBias')
            dataSubj.pe_BayesianAgent_confirmBias = peSims;
            dataSubj.up_BayesianAgent_confirmBias = upSims;
            dataSubj.lr_pe_BayesianAgent_confirmBias = lrSims;
            data_bayesianAgent_confirmBias = [data_bayesianAgent_confirmBias; dataSubj];
        end

    end


    hold on
    plot(nanmean(mix_sims_all))
    hold on
    shadedErrorBar(1:25,nanmean(mix_sims_all),nanstd(mix_sims_all)./sqrt(numSubjs), ...
        {'LineWidth', 2,'Color',colors(m,:)},1)
    hold on
    plot(nanmean(perc_sims_all))
    hold on
    shadedErrorBar(1:25,nanmean(perc_sims_all),nanstd(perc_sims_all)./sqrt(numSubjs), ...
        {'LineWidth', 2,'Color',colors(m,:)},1)

end
% directory specification
% Get the current working directory
currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"main");
mkdir(save_dir)

% load all required data
% Base directory for data files
base_dir = strcat(desiredPath, filesep, 'Data');

mean_curves = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'mean_curves.mat')); % learning across trials for each condition
sem_curves = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'sem_curves.mat'));

mix_curve = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'mix_curve.mat')); % learning in a block for each condition
perc_curve = importdata(strcat(base_dir, filesep, 'descriptive data', filesep, 'main study', filesep, 'perc_curve.mat'));

hold on
plot(nanmean(mix_curve))
hold on
shadedErrorBar(1:25,nanmean(mix_curve),nanstd(mix_curve)./sqrt(numSubjs),{'LineWidth', 2,'Color',[0.5,0.5,0.5]},1)
hold on
plot(nanmean(perc_curve))
hold on
shadedErrorBar(1:25,nanmean(perc_curve),nanstd(perc_curve)./sqrt(numSubjs), ...
    {'LineWidth', 2,'Color',[0.5,0.5,0.5]},1)
legend({'','Basic RL','','','RL Sigma','','','','RL Sigma + CB','','','','Bayesian agent','','','','Bayesian agent + CB'})

%%

% Create figure with two subplots
fig = figure;
colors = lines(5);
model_names = {'basicRL','RLEstSens','RLEstSens_confirmBias','BayesianAgent','BayesianAgent_confirmBias'};

% Initialize storage for condition-specific data
mix_sims_all_models = NaN(numModels, numSubjs, 25);
perc_sims_all_models = NaN(numModels, numSubjs, 25);

% Loop through models and collect data
for m = 1:numModels
    for n = 1:numSubjs
        for ns = 1:nSimulations
            dataSubj = data(data.ID == uniqueID(n),:);
            valid = ~isnan(dataSubj.mu_congruence);
            dataSubj = dataSubj(valid, :);
            
            % Get model-specific mu data
            if ismember(model_names{m},'basicRL')
                mu_data_subj = mu_data.basicRL{n,1};
            elseif ismember(model_names{m},'RLEstSens')
                mu_data_subj = mu_data.RLEstSens{n,1};
            elseif ismember(model_names{m},'RLEstSens_confirmBias')
                mu_data_subj = mu_data.RLEstSens_confirmBias{n,1};
            elseif ismember(model_names{m},'BayesianAgent')
                mu_data_subj = mu_data.BayesianAgent{n,1};
            elseif ismember(model_names{m},'BayesianAgent_confirmBias')
                mu_data_subj = mu_data.BayesianAgent_confirmBias{n,1};
            end
            
            dataSubj.mu_model = mu_data_subj(:,ns);
            
            % Transform mu based on contrast
            for i = 2:height(dataSubj)
                if dataSubj.contrast(i) == 1
                    dataSubj.mu_t_1_model(i) = 1-dataSubj.mu_model(i-1);
                    dataSubj.mu_t_model(i) = 1-dataSubj.mu_model(i);
                else
                    dataSubj.mu_t_1_model(i) = dataSubj.mu_model(i-1);
                    dataSubj.mu_t_model(i) = dataSubj.mu_model(i);
                end
            end
            
            % Extract blocks for each condition
            uni_mix = unique(dataSubj.blocks(dataSubj.choice_cond==1));
            uni_perc = unique(dataSubj.blocks(dataSubj.choice_cond==2));
            
            mix_subj_blocks = NaN(num_blocks, t);
            perc_subj_blocks = NaN(num_blocks, t);
            
            for b = 1:num_blocks
                temp_mix = dataSubj.mu_model(and(dataSubj.blocks == uni_mix(b), dataSubj.choice_cond == 1));
                temp_perc = dataSubj.mu_model(and(dataSubj.blocks == uni_perc(b), dataSubj.choice_cond == 2));
                
                % Pad or truncate to 25 trials
                if length(temp_mix) < t
                    temp_mix(end+1:t) = NaN;
                elseif length(temp_mix) > t
                    temp_mix = temp_mix(1:t);
                end
                
                if length(temp_perc) < t
                    temp_perc(end+1:t) = NaN;
                elseif length(temp_perc) > t
                    temp_perc = temp_perc(1:t);
                end
                
                mix_subj_blocks(b,:) = temp_mix(:)';
                perc_subj_blocks(b,:) = temp_perc(:)';
            end
            
            mix_sims(ns,:) = mean(mix_subj_blocks, 'omitnan');
            perc_sims(ns,:) = mean(perc_subj_blocks, 'omitnan');
        end
        
        mix_sims_all(n,:) = mean(mix_sims, 'omitnan');
        perc_sims_all(n,:) = mean(perc_sims, 'omitnan');
    end
    
    % Store data for each model
    mix_sims_all_models(m,:,:) = mix_sims_all;
    perc_sims_all_models(m,:,:) = perc_sims_all;
end

% Create subplots
subplot(1,2,1);
hold on;
for m = 1:numModels
    mix_mean = squeeze(nanmean(mix_sims_all_models(m,:,:), 2));
    mix_sem = squeeze(nanstd(mix_sims_all_models(m,:,:), [], 2)) ./ sqrt(numSubjs);
    
    % Plot mean line
    plot(1:25, mix_mean, 'Color', colors(m,:), 'LineWidth', 2);
    
    % Add shaded error bar
    shadedErrorBar(1:25, mix_mean, mix_sem, ...
        {'LineWidth', 2, 'Color', colors(m,:)}, 1);
end

% Add empirical data for mix condition
plot(1:25, nanmean(mix_curve), 'Color', [0.5,0.5,0.5], 'LineWidth', 2);
shadedErrorBar(1:25, nanmean(mix_curve), nanstd(mix_curve)./sqrt(numSubjs), ...
    {'LineWidth', 2, 'Color', [0.5,0.5,0.5]}, 1);

title('Mix Condition (70% probability)');
xlabel('Trial');
ylabel('Mean μ');
ylim([0.4, 1.0]);
legend([model_names, 'Empirical'], 'Location', 'best');
hold off;

subplot(1,2,2);
hold on;
for m = 1:numModels
    perc_mean = squeeze(nanmean(perc_sims_all_models(m,:,:), 2));
    perc_sem = squeeze(nanstd(perc_sims_all_models(m,:,:), [], 2)) ./ sqrt(numSubjs);
    
    % Plot mean line
    plot(1:25, perc_mean, 'Color', colors(m,:), 'LineWidth', 2);
    
    % Add shaded error bar
    shadedErrorBar(1:25, perc_mean, perc_sem, ...
        {'LineWidth', 2, 'Color', colors(m,:)}, 1);
end

% Add empirical data for perc condition
plot(1:25, nanmean(perc_curve), 'Color', [0.5,0.5,0.5], 'LineWidth', 2);
shadedErrorBar(1:25, nanmean(perc_curve), nanstd(perc_curve)./sqrt(numSubjs), ...
    {'LineWidth', 2, 'Color', [0.5,0.5,0.5]}, 1);

title('Perc Condition (90% probability)');
xlabel('Trial');
ylabel('Mean μ');
ylim([0.4, 1.0]);
legend([model_names, 'Empirical'], 'Location', 'best');
hold off;

% Adjust figure properties
set(fig, 'Position', [100, 100, 1200, 500]);
sgtitle('Learning Curves by Condition');

% Save figure
% save_path = fullfile(save_dir, 'learning_curves_by_condition.fig');
% savefig(fig, save_path);
% save_path_png = fullfile(save_dir, 'learning_curves_by_condition.png');
% saveas(fig, save_path_png);


%% PLOT PE vs. UP

pe_edges = linspace(0,1,11);
binned_data = abs(data_basicRL.pe_basicRL);
data_basicRL.pe_bins = discretize(abs(data_basicRL.pe_basicRL),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_basicRL.ID(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
y_data = abs(data_basicRL.up_basicRL(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2));
bins = data_basicRL.pe_bins(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
binned_data = binned_data(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
figure
hold on
subplot(2,3,1)
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(1,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,1,11);
binned_data = abs(data_RLSigma.pe_RLSigma);
data_RLSigma.pe_bins = discretize(abs(data_RLSigma.pe_RLSigma),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_RLSigma.ID(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
y_data = abs(data_RLSigma.up_RLSigma(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2));
bins = data_RLSigma.pe_bins(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
binned_data = binned_data(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,2)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(2,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,1,11);
binned_data = abs(dataRLSigma_confirmBias.pe_RLSigma_confirmBias);
dataRLSigma_confirmBias.pe_bins = discretize(abs(dataRLSigma_confirmBias.pe_RLSigma_confirmBias),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = dataRLSigma_confirmBias.ID(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
y_data = abs(dataRLSigma_confirmBias.up_RLSigma_confirmBias(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2));
bins = dataRLSigma_confirmBias.pe_bins(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
binned_data = binned_data(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,3)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(3,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,1,11);
binned_data = abs(data_bayesianAgent.pe_BayesianAgent);
data_bayesianAgent.pe_bins = discretize(abs(data_bayesianAgent.pe_BayesianAgent),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent.ID(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
y_data = abs(data_bayesianAgent.up_BayesianAgent(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2));
bins = data_bayesianAgent.pe_bins(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
binned_data = binned_data(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,4)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(4,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,1,11);
binned_data = abs(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias);
data_bayesianAgent_confirmBias.pe_bins = discretize(abs(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent_confirmBias.ID(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
y_data = abs(data_bayesianAgent_confirmBias.up_BayesianAgent_confirmBias(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2));
bins = data_bayesianAgent_confirmBias.pe_bins(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
binned_data = binned_data(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,5)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(5,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,1,11);
binned_data = abs(data.pe);
data.pe_bins = discretize(abs(data.pe),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
data.lr = data.up./data.pe;
run_id = data.ID(data.pe ~= 0 & abs(data.lr)<=2);
y_data = abs(data.up(data.pe ~= 0 & abs(data.lr)<=2));
bins = data.pe_bins(data.pe ~= 0 & abs(data.lr)<=2);
binned_data = binned_data(data.pe ~= 0 & abs(data.lr)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,6)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',[0.5,0.5,0.5]);
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')


%% PLOT Condiff vs. Abs LR

data_basicRL.con_diff_choice = abs(data_basicRL.contrast_left - data_basicRL.contrast_right)./2;
data_RLSigma.con_diff_choice = abs(data_RLSigma.contrast_left - data_RLSigma.contrast_right)./2;
dataRLSigma_confirmBias.con_diff_choice = abs(dataRLSigma_confirmBias.contrast_left - dataRLSigma_confirmBias.contrast_right)./2;
data_bayesianAgent.con_diff_choice = abs(data_bayesianAgent.contrast_left - data_bayesianAgent.contrast_right)./2;
data_bayesianAgent_confirmBias.con_diff_choice = abs(data_bayesianAgent_confirmBias.contrast_left - data_bayesianAgent_confirmBias.contrast_right)./2;

figure
hold on

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_basicRL.con_diff_choice);
data_basicRL.pe_bins = discretize(abs(data_basicRL.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_basicRL.ID(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
y_data = abs(data_basicRL.lr_basicRL(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2));
bins = data_basicRL.pe_bins(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
binned_data = binned_data(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,1)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(1,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_RLSigma.con_diff_choice);
data_RLSigma.pe_bins = discretize(abs(data_RLSigma.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_RLSigma.ID(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
y_data = abs(data_RLSigma.lr_RLSigma(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2));
bins = data_RLSigma.pe_bins(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
binned_data = binned_data(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,2)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(2,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')


pe_edges = linspace(0,0.1,11);
binned_data = abs(dataRLSigma_confirmBias.con_diff_choice);
dataRLSigma_confirmBias.pe_bins = discretize(abs(dataRLSigma_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = dataRLSigma_confirmBias.ID(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
y_data = abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2));
bins = dataRLSigma_confirmBias.pe_bins(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
binned_data = binned_data(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,3)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(3,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_bayesianAgent.con_diff_choice);
data_bayesianAgent.pe_bins = discretize(abs(data_bayesianAgent.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent.ID(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
y_data = abs(data_bayesianAgent.lr_BayesianAgent(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2));
bins = data_bayesianAgent.pe_bins(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
binned_data = binned_data(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT

subplot(2,3,4)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(4,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_bayesianAgent_confirmBias.con_diff_choice);
data_bayesianAgent_confirmBias.pe_bins = discretize(abs(data_bayesianAgent_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent_confirmBias.ID(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
y_data = abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2));
bins = data_bayesianAgent_confirmBias.pe_bins(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
binned_data = binned_data(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,5)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(5,:));
xlabel('PE bins (1 bin = 0.1)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data.con_diff_choice);
data.pe_bins = discretize(abs(data.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
data.lr = data.up./data.pe;
run_id = data.ID(data.pe ~= 0 & abs(data.lr)<=2);
y_data = abs(data.lr(data.pe ~= 0 & abs(data.lr)<=2));
bins = data.pe_bins(data.pe ~= 0 & abs(data.lr)<=2);
binned_data = binned_data(data.pe ~= 0 & abs(data.lr)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,6)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',[0.5,0.5,0.5]);
xlabel('Condiff bins (1 bin = 0.01)')
ylabel('Mean absolute learning rate (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

%% PLOT CONDIFF vs. Abs UP

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_basicRL.con_diff_choice);
data_basicRL.pe_bins = discretize(abs(data_basicRL.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_basicRL.ID(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
y_data = abs(data_basicRL.up_basicRL(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2));
bins = data_basicRL.pe_bins(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
binned_data = binned_data(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
figure
hold on
subplot(2,3,1)
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(1,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_RLSigma.con_diff_choice);
data_RLSigma.pe_bins = discretize(abs(data_RLSigma.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_RLSigma.ID(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
y_data = abs(data_RLSigma.up_RLSigma(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2));
bins = data_RLSigma.pe_bins(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
binned_data = binned_data(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,2)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(2,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(dataRLSigma_confirmBias.con_diff_choice);
dataRLSigma_confirmBias.pe_bins = discretize(abs(dataRLSigma_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = dataRLSigma_confirmBias.ID(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
y_data = abs(dataRLSigma_confirmBias.up_RLSigma_confirmBias(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2));
bins = dataRLSigma_confirmBias.pe_bins(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
binned_data = binned_data(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,3)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
% ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(3,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_bayesianAgent.con_diff_choice);
data_bayesianAgent.pe_bins = discretize(abs(data_bayesianAgent.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent.ID(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
y_data = abs(data_bayesianAgent.up_BayesianAgent(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2));
bins = data_bayesianAgent.pe_bins(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
binned_data = binned_data(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT

subplot(2,3,4)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(4,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_bayesianAgent_confirmBias.con_diff_choice);
data_bayesianAgent_confirmBias.pe_bins = discretize(abs(data_bayesianAgent_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent_confirmBias.ID(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
y_data = abs(data_bayesianAgent_confirmBias.up_BayesianAgent_confirmBias(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2));
bins = data_bayesianAgent_confirmBias.pe_bins(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
binned_data = binned_data(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,5)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(5,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean absolute LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')


pe_edges = linspace(0,0.1,11);
binned_data = abs(data.con_diff_choice);
data.pe_bins = discretize(abs(data.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
data.lr = data.up./data.pe;
run_id = data.ID(data.pe ~= 0 & abs(data.lr)<=2);
y_data = abs(data.up(data.pe ~= 0 & abs(data.lr)<=2));
bins = data.pe_bins(data.pe ~= 0 & abs(data.lr)<=2);
binned_data = binned_data(data.pe ~= 0 & abs(data.lr)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,6)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',[0.5,0.5,0.5]);
xlabel('Condiff bins (1 bin = 0.01)')
ylabel('Mean absolute updates (UP)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

%% PLOT CONDIFF vs. Signed LR

pe_edges = linspace(0,0.1,11);
binned_data = abs(data_basicRL.con_diff_choice);
data_basicRL.pe_bins = discretize(abs(data_basicRL.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
data_basicRL.lr_basicRL = data_basicRL.up_basicRL./data_basicRL.pe_basicRL;
run_id = data_basicRL.ID(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
y_data = data_basicRL.lr_basicRL(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
bins = data_basicRL.pe_bins(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);
binned_data = binned_data(data_basicRL.pe_basicRL ~= 0 & abs(data_basicRL.lr_basicRL)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
figure
subplot(2,3,1)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(1,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
data_RLSigma.lr_RLSigma = data_RLSigma.up_RLSigma./data_RLSigma.pe_RLSigma;
binned_data = abs(data_RLSigma.con_diff_choice);
data_RLSigma.pe_bins = discretize(abs(data_RLSigma.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_RLSigma.ID(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
y_data = data_RLSigma.lr_RLSigma(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
bins = data_RLSigma.pe_bins(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);
binned_data = binned_data(data_RLSigma.pe_RLSigma ~= 0 & abs(data_RLSigma.lr_RLSigma)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,2)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(2,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
dataRLSigma_confirmBias.lr_RLSigma_confirmBias = dataRLSigma_confirmBias.up_RLSigma_confirmBias./dataRLSigma_confirmBias.pe_RLSigma_confirmBias;
binned_data = abs(dataRLSigma_confirmBias.con_diff_choice);
dataRLSigma_confirmBias.pe_bins = discretize(abs(dataRLSigma_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = dataRLSigma_confirmBias.ID(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
y_data = dataRLSigma_confirmBias.lr_RLSigma_confirmBias(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
bins = dataRLSigma_confirmBias.pe_bins(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);
binned_data = binned_data(dataRLSigma_confirmBias.pe_RLSigma_confirmBias ~= 0 & abs(dataRLSigma_confirmBias.lr_RLSigma_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT

subplot(2,3,3)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
ls.Color = 'k';
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(3,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
data_bayesianAgent.lr_BayesianAgent = data_bayesianAgent.up_BayesianAgent./data_bayesianAgent.pe_BayesianAgent;
binned_data = abs(data_bayesianAgent.con_diff_choice);
data_bayesianAgent.pe_bins = discretize(abs(data_bayesianAgent.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent.ID(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
y_data = data_bayesianAgent.lr_BayesianAgent(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
bins = data_bayesianAgent.pe_bins(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);
binned_data = binned_data(data_bayesianAgent.pe_BayesianAgent ~= 0 & abs(data_bayesianAgent.lr_BayesianAgent)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT

subplot(2,3,4)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(4,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

pe_edges = linspace(0,0.1,11);
data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias = data_bayesianAgent_confirmBias.up_BayesianAgent_confirmBias./data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias;
binned_data = abs(data_bayesianAgent_confirmBias.con_diff_choice);
data_bayesianAgent_confirmBias.pe_bins = discretize(abs(data_bayesianAgent_confirmBias.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
run_id = data_bayesianAgent_confirmBias.ID(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
y_data = data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
bins = data_bayesianAgent_confirmBias.pe_bins(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);
binned_data = binned_data(data_bayesianAgent_confirmBias.pe_BayesianAgent_confirmBias ~= 0 & abs(data_bayesianAgent_confirmBias.lr_pe_BayesianAgent_confirmBias)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,5)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',colors(5,:));
xlabel('Condiff bins (1 bin = 0.1)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')


pe_edges = linspace(0,0.1,11);
binned_data = abs(data.con_diff_choice);
data.pe_bins = discretize(abs(data.con_diff_choice),pe_edges);
nbins = 10; % number of bins

% GET RID OF TRIALS WHERE PE = 0 AND OUTLIER LRs
data.lr = data.up./data.pe;
run_id = data.ID(data.pe ~= 0 & abs(data.lr)<=2);
y_data = data.lr(data.pe ~= 0 & abs(data.lr)<=2);
bins = data.pe_bins(data.pe ~= 0 & abs(data.lr)<=2);
binned_data = binned_data(data.pe ~= 0 & abs(data.lr)<=2);

% MEAN LRs for CONDIFF BINS
avg_ydata_bins = NaN(nbins,numSubjs); 
avg_behv_bins = NaN(nbins,numSubjs); 
for b = 1:nbins % run for each bin
    for n = 1:numSubjs % run for each subject
        bins_subj = bins(run_id == uniqueID(n)); 
        y_data_subj = y_data(run_id == uniqueID(n)); % data from that bin for that subject
        binned_data_subj = binned_data(run_id == uniqueID(n)); % binned data for that subject
        avg_behv_bins(b,n) = nanmean(binned_data_subj(bins_subj == b)); 
        avg_ydata_bins(b,n) = nanmean(y_data_subj(bins_subj == b)); % mean of that data within that bin per subject
    end
end
avg_ydata = nanmean(avg_ydata_bins,2); 
avg_binneddata = nanmean(avg_behv_bins,2);
sem_ydata = nanstd(avg_ydata_bins,0,2)./sqrt(numSubjs);

% PLOT
subplot(2,3,6)
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor',"none",'MarkerFaceColor',"none");
hold on
ls = lsline;
hold('on')
errorbar(1:nbins,avg_ydata, sem_ydata, 'k', 'LineWidth',1,'LineStyle','none');
hold on
s1 = scatter(1:nbins,avg_ydata,"filled",'MarkerEdgeColor','k','MarkerFaceColor',[0.5,0.5,0.5]);
xlabel('Condiff bins (1 bin = 0.01)')
ylabel('Mean signed LR (LR)')
title(strcat("\itr\rm = ",{''},num2str(round(corr((1:nbins).',avg_ydata),2))),'FontWeight','normal')

%% PLOT EST. ERROR CURVES


%%

for i = 2:height(dataSubj)
    if dataSubj.contrast(i) == 1 % if actual mu < 0.5
        dataSubj.mu_t_1_basicRL(i) = 1-dataSubj.mu_basicRL(i-1);
        dataSubj.mu_t_basicRL(i) = 1-dataSubj.mu_basicRL(i);
    else
        dataSubj.mu_t_1_basicRL(i) = dataSubj.mu_basicRL(i-1);
        dataSubj.mu_t_basicRL(i) = dataSubj.mu_basicRL(i);
    end
end

state_zero_idx = dataSubj.state == 0; % index for rows where state is 0

% COMPUTE PE
dataSubj.pe_basicRL(state_zero_idx) = dataSubj.recoded_reward(state_zero_idx) - dataSubj.mu_t_1_basicRL(state_zero_idx); % state = 0
dataSubj.pe_basicRL(~state_zero_idx) = (1 - dataSubj.recoded_reward(~state_zero_idx)) - dataSubj.mu_t_1_basicRL(~state_zero_idx); % state = 1

% COMPUTE UP
dataSubj.up_basicRL = NaN(height(dataSubj),1);
dataSubj.up_basicRL(2:end) = dataSubj.mu_t_basicRL(2:height(dataSubj)) - dataSubj.mu_t_1_basicRL(2:height(dataSubj));
dataSubj.pe_basicRL(dataSubj.trials == 1,1) = 0;

% for h = 1:height(dataAll)-1
%     dataAll.up_basicRL(h) = dataAll.mu_basicRL(h+1) - dataAll.mu_basicRL(h);
%     dataAll.pe_basicRL(h) = dataAll.recoded_reward(h+1) - dataAll.mu_basicRL(h);
% end

% safe_saveall('preprocessed_withRLSigmaSims_sampling.xlsx',dataAll);

%% COMPUTE SUBJECTIVE CONTINGENCY PARAMETER FOR INCONGRUENT BLOCKS

% INITIALISE TO STORE MEAN MU FOR ALL TRIALS, ACROSS SUBJECTS
t = 25;
mix_curve = NaN(length(uniqueID),t);
perc_curve = NaN(length(uniqueID),t);
num_blocks = 4;

for i = [1:numSubjs]
    data_subj_choice = dataSubj(dataSubj.ID==uniqueID(i),:); % for each subject
    uni_mix = unique(data_subj_choice.blocks(data_subj_choice.choice_cond==1)); % block number for condition = 1
    uni_perc = unique(data_subj_choice.blocks(data_subj_choice.choice_cond==2)); % block number for condition = 2
    mix_subj = NaN(num_blocks,t);
    perc_subj = NaN(num_blocks,t);
    for b = 1:num_blocks
        temp_mix = data_subj_choice.mu_basicRL(and(data_subj_choice.blocks == uni_mix(b), data_subj_choice.choice_cond == 1));
        temp_perc = data_subj_choice.mu_basicRL(and(data_subj_choice.blocks == uni_perc(b), data_subj_choice.choice_cond == 2));

        % Pad with NaN if fewer than 25 trials, or truncate if more
        if length(temp_mix) < t
            temp_mix(end+1:t) = NaN;
        elseif length(temp_mix) > t
            temp_mix = temp_mix(1:t);
        end

        if length(temp_perc) < t
            temp_perc(end+1:t) = NaN;
        elseif length(temp_perc) > t
            temp_perc = temp_perc(1:t);
        end

        mix_subj(b,:) = temp_mix(:)';  % Ensure row vector
        perc_subj(b,:) = temp_perc(:)'; % Ensure row vector
    end
    mix_curve(i,:) = mean(mix_subj);
    perc_curve(i,:) = mean(perc_subj);
end

mean_curves = [nanmean(mix_curve);nanmean(perc_curve);];
sem_curves = [nanstd(mix_curve)./sqrt(numSubjs);nanstd(perc_curve)./sqrt(numSubjs);];


%%

preprocess_obj = preprocess_LR(); % initialise object with all required variables and functions
% preprocess_obj.flip_mu(); % compute reported contingency parameter, after correcting for congruence
preprocess_obj.compute_action_dep_rew(); % compute action dependent reward
% preprocess_obj.compute_mu(); % recode mu, contingent on if actual mu < 0.5 or not
preprocess_obj.compute_state_dep_pe(); % compute state dependent PE and UP

% USER-BASED PATH
currentDir = cd; % current directory
reqPath = 'Perceptual_unc_aug_task_pupil-main'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = "/Users/prashantig/Brown Dropbox/Prashanti Ganesh/PhD/Semester 8/pupil_manuscript/" + ...
    "Perceptual_unc_aug_task_pupil-main/NatCommns Revisions/Reviewer 2/behavior/model fitting";
mkdir(save_dir);

% SAVE PREPROCESSED FILE
safe_saveall(fullfile(save_dir,'preprocessed_withModelFittingSims_PE.xlsx'),preprocess_obj.data)
safe_saveall(fullfile(save_dir,'preprocessed_withModelFittingSims_PE_no_zerope.xlsx'),preprocess_obj.data(preprocess_obj.data.pe ~= 0, :));
% simulate data - mu curves, pe, up
% plot mu curves
% plot pe vs. up
% plot condiff vs. up
