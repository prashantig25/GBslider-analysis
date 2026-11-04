%% ========================================================================
%  Script: Analytical Belief Update, With vs Without Integration, Long Block
%  ------------------------------------------------------------------------
%  Analytical (eval_ana=1) sibling of simulateAndPlot_numerical_
%  intVsNoInt_longBlock.m: simulates ONE long (40-trial) block per
%  (sigma, true mu) combination, using the Agent's analytical (eval_ana=1,
%  polynomial) belief update, once WITH integration
%  (task_agent_analysis=1) and once WITHOUT (task_agent_analysis=0).
%  o_t=c_t exactly for both (task_agent_analysis controls this too -- see
%  Agent.m's observation_sample).
%  Plots mean +/- SEM simulated-mu curves plus thin individual-simulation
%  lines, in a true-mu (0.7, 0.9) x sigma subplot grid -- same style as
%  Agent/simfigures/simulatedMu_withVsNoInt_evalAna1.pdf (produced by
%  plotLongBlocks_integrationComparison.m), same mu (0.7, 0.9) and sigma
%  ([0.02, 0.04, 0.06, 0.08, 0.1]) levels as this script's numerical
%  sibling.
%  ------------------------------------------------------------------------
%  Design notes:
%  - num_trials_long=40, not 1000 -- eval_ana=1's polynomial belief update
%    becomes numerically unstable past ~50-75 trials (see script_sims.m's
%    own eval_ana=1 section header), so 40 is the longest block length
%    used elsewhere in this repo for the analytical solution.
%  - mu=0.7 -> Task/taskvars condition 1; mu=0.9 -> condition 7 (see
%    Agent/Task.m's taskMu) -- same conditions as the numerical sibling
%    script, not the condition 1/2 (0.7/1) pairing used in script_sims.m's
%    own long-block sections.
%  - Plain for-loop (not parfor) over task_agent_analysis -- same reason
%    as the numerical sibling script: this script keeps data_withInt/
%    data_noInt in memory for in-script plotting, which parfor's variable-
%    classification rules don't handle well for this conditional-
%    assignment pattern.
%  - Data is saved via safe_saveall before plotting, so this script's
%    simulation doesn't need to be re-run just to re-plot -- comment out
%    the %% SIMULATE section and load the saved files with readtable
%    instead if re-plotting only.
%  ========================================================================
clc; clearvars; close all;

% PATH STUFF
currentDir = cd;
reqPath = 'GBslider-analysis';
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'Data', 'agent simulations');
save_fig_dir = fullfile(desiredPath, 'Agent', 'simfigures');
if ~exist(save_fig_dir, 'dir')
    mkdir(save_fig_dir);
end

%% PARAMETERS
num_trials_long = 40;  % trials in the single long block -- capped for eval_ana=1's numerical stability, see design notes
num_blocks_long = 1;   % one long block per simulation
num_sims_long = 50;
sigma_values_long = [0.02, 0.04, 0.06, 0.08, 0.1];
conditions_long = [1, 7]; % condition 1 -> true mu 0.7, condition 7 -> true mu 0.9 (Agent/Task.m's taskMu)
true_mu_by_cond = [0.7, 0.9];
beta_long = 100; % Agent's own default (agentvars.m)
agent_long = 1;  % normative agent
eval_ana_long = 1; % analytical (polynomial) belief update

%% SIMULATE
for task_agent_analysis_long = [1, 0]
    simulations_long = [];
    for sigma_long = sigma_values_long
        for condition_long = conditions_long
            sim_chunk = [];
            [sim_chunk] = run_simulations(num_sims_long,num_blocks_long,sim_chunk,condition_long,agent_long,0,task_agent_analysis_long,sigma_long,beta_long,num_trials_long,eval_ana_long); % use run simulations function
            sim_chunk.sigma = repmat(sigma_long,height(sim_chunk),1);
            sim_chunk.choice_cond = repmat(condition_long,height(sim_chunk),1);
            simulations_long = [simulations_long; sim_chunk];
        end
    end
    if task_agent_analysis_long == 1
        data_withInt = simulations_long;
        safe_saveall(fullfile(save_dir,'data_agent_analytical_longBlock40_withInt.txt'),simulations_long);
    else
        data_noInt = simulations_long;
        safe_saveall(fullfile(save_dir,'data_agent_analytical_longBlock40_noInt.txt'),simulations_long);
    end
end

%% PLOT
font_size = 9;
font_name = 'Arial';
line_width_mean = 1.5; % mean-curve line width
line_width_thin = 0.05; % single-simulation trace line width
noIntegration_color = [0, 0.4470, 0.7410]; % MATLAB default blue
noIntegration_color_light = noIntegration_color + (1 - noIntegration_color) * 0.75; % lightened (towards white), individual-simulation lines
integration_color = [0.8500, 0.3250, 0.0980]; % MATLAB default orange
integration_color_light = integration_color + (1 - integration_color) * 0.75; % lightened (towards white), individual-simulation lines

fig = figure('Position', [50, 50, 1500, 500]);
plot_idx = 1;
for c = 1:length(conditions_long)
    cond = conditions_long(c);
    for s = 1:length(sigma_values_long)
        sigma_val = sigma_values_long(s);
        ax = subplot(length(conditions_long), length(sigma_values_long), plot_idx);
        hold on;

        values_noInt = local_trial_values(data_noInt, cond, sigma_val, 'mu', num_trials_long);
        mean_noInt = mean(values_noInt, 2);
        sem_noInt = std(values_noInt, 0, 2) ./ sqrt(size(values_noInt, 2));

        values_withInt = local_trial_values(data_withInt, cond, sigma_val, 'mu', num_trials_long);
        mean_withInt = mean(values_withInt, 2);
        sem_withInt = std(values_withInt, 0, 2) ./ sqrt(size(values_withInt, 2));

        % Individual-simulation lines drawn first so the mean+SEM overlays sit on top.
        plot(1:num_trials_long, values_noInt, 'Color', noIntegration_color_light, 'LineWidth', line_width_thin);
        plot(1:num_trials_long, values_withInt, 'Color', integration_color_light, 'LineWidth', line_width_thin);
        h_noInt = shadedErrorBar(1:num_trials_long, mean_noInt, sem_noInt, {'Color', noIntegration_color, 'LineWidth', line_width_mean}, 0);
        h_withInt = shadedErrorBar(1:num_trials_long, mean_withInt, sem_withInt, {'Color', integration_color, 'LineWidth', line_width_mean}, 0);
        h_trueMu = yline(true_mu_by_cond(c), '--k', 'LineWidth', 1);

        title(sprintf('mu=%.1f, sigma=%.2f', true_mu_by_cond(c), sigma_val), 'FontWeight', 'normal', 'FontSize', font_size);
        if c == length(conditions_long)
            xlabel('Trial');
        end
        if s == 1
            ylabel('Simulated mu');
        end
        set(ax, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1, 'Color', 'none');
        box off;
        ylim([0, 1]);
        plot_idx = plot_idx + 1;
    end
end
legend([h_noInt.mainLine, h_withInt.mainLine, h_trueMu], {'No Integration', 'Integration', 'True mu'}, 'Position', [0.92, 0.45, 0.07, 0.1], 'Box', 'off', 'Color', 'none');
sgtitle('Simulated mu: Analytical Belief Update, With vs Without Integration (eval\_ana=1, o\_t=c\_t)', 'FontWeight', 'bold');

exportgraphics(fig, fullfile(save_fig_dir, 'simulatedMu_analytical_withVsNoInt_longBlock40.pdf'), 'ContentType', 'vector');

%% ========================================================================
%  Local functions
%  ========================================================================
function values_matrix = local_trial_values(data, cond, sigma_val, varname, num_trials_long)
    subset = data(data.choice_cond == cond & abs(data.sigma - sigma_val) < 1e-9, :);
    num_sims = height(subset) / num_trials_long;
    values_matrix = reshape(subset.(varname), num_trials_long, num_sims); % trials x sims
end
