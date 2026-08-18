%% Family-wise Model Comparison using VBA Toolbox - FIXED VERSION
% This addresses the model fragmentation issue by grouping related models
clc
clearvars
AICBIC_table = importdata('AICBIC_RBVoi_integratedBoth.mat');

%% 1. Prepare Log Model Evidence Matrix
% Convert from information criteria to log model evidence
BIC_matrix = [AICBIC_table.BIC_basicRL, ...
                  AICBIC_table.BIC_RLsigma, ...
                  AICBIC_table.BIC_RLSigma_confirmBias, ...
                  AICBIC_table.BIC_BayesianAgent, ...
                  AICBIC_table.BIC_BayesianAgent_confirmBias,...
                  AICBIC_table.BIC_basicRL_confirmBias];
lme = -0.5 * BIC_matrix;  % Convert BIC to log model evidence

% Check dimensions
[n_subjects, n_models] = size(lme);
fprintf('Data: %d subjects, %d models\n', n_subjects, n_models);

%% 2. Define Model Families
family_names = {'Basic_RL', 'LR_Modulation'};
% Define which models belong to which family
families{1} = [1,6];           % BasicRL (models 1 and 6)
families{2} = [2, 3, 4, 5];    % LR modulation models (2-5)

%% 3. Run Family-wise Bayesian Model Selection
fprintf('Running family-wise model comparison...\n');
% Set up VBA options structure
options_vba = struct();
options_vba.families = families;
options_vba.verbose = 1;
options_vba.DisplayWin = 0;

try
    % VBA_groupBMC with proper options structure
    [posterior, out] = VBA_groupBMC(lme', options_vba);
    fprintf('VBA_groupBMC completed successfully!\n');
    vba_success = true;
catch ME
    fprintf('VBA_groupBMC failed with error: %s\n', ME.message);
    fprintf('Using manual implementation instead...\n');
    [posterior, out] = manual_family_BMS(lme, families, family_names);
    vba_success = false;
end

%% 4. Extract and Display Results - FIXED VERSION
if vba_success
    % VBA succeeded - extract results correctly
    family_prob = out.families.Ef;        % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = out.Ef;                  % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities
    
    fprintf('\n=== VBA FAMILY-WISE RESULTS ===\n');
else
    % Manual implementation results
    family_prob = posterior.r;            % Family probabilities
    family_xp = out.families.ep;          % Family exceedance probabilities
    model_prob = posterior.r_model;       % Individual model probabilities
    model_xp = out.ep;                    % Individual model exceedance probabilities
    
    fprintf('\n=== MANUAL FAMILY-WISE RESULTS ===\n');
end

% Display family results
for i = 1:length(families)
    fprintf('Family %d (%s):\n', i, family_names{i});
    fprintf('  Probability: %.3f\n', family_prob(i));
    fprintf('  Exceedance Probability: %.3f\n', family_xp(i));
    fprintf('  Models included: %s\n', mat2str(families{i}));
    fprintf('\n');
end

% Display model results
fprintf('=== MODEL-LEVEL RESULTS ===\n');
model_names = {'BasicRL', 'RLSigma', 'RLSigma + CS', 'Agent', 'Agent + CS', 'BasicRL + CS'};
for i = 1:n_models
    fprintf('Model %d (%s): Prob=%.3f, XP=%.3f\n', i, model_names{i}, model_prob(i), model_xp(i));
end

%% 5. Enhanced Visualization with Copper Colormap
% Create copper colormap colors with alpha
copper_colors = copper(length(families) + n_models);
colors.dark_gray = [0.3, 0.3, 0.3];  % Dark gray for text

% Create main figure
fig = figure('Position', [100, 100, 1400, 500], 'Color', 'white');

%% Plot 1: Family Probabilities
subplot(1, 3, 1);
h1 = bar(family_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

% Set copper colormap colors for families
h1.CData(1,:) = copper_colors(1, 1:3);  % First copper color
h1.CData(2,:) = copper_colors(2, 1:3);  % Second copper color

% Styling - remove grid and use normal font
ylim([0, 1]);
ylabel('Family Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', colors.dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels on bars
for i = 1:length(family_prob)
    text(i, family_prob(i) + 0.03, sprintf('%.3f', family_prob(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', colors.dark_gray);
end

ax1 = gca;
ax1.Color = [0.98, 0.98, 0.98];

%% Plot 2: Family Exceedance Probabilities  
subplot(1, 3, 2);
h2 = bar(family_xp, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.6, 'FaceAlpha', 0.3);

% Set copper colormap colors for families (slightly darker)
h2.CData(1,:) = copper_colors(1, 1:3) * 0.8;
h2.CData(2,:) = copper_colors(2, 1:3) * 0.8;

% Styling - remove grid and use normal font
ylim([0, 1]);
ylabel('Exceedance Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Family Exceedance Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', colors.dark_gray);
set(gca, 'XTickLabel', {'Basic RL', 'LR Modulation'}, 'FontSize', 11, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels
for i = 1:length(family_xp)
    text(i, family_xp(i) + 0.03, sprintf('%.3f', family_xp(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
         'FontSize', 11, 'Color', colors.dark_gray);
end

ax2 = gca;
ax2.Color = [0.98, 0.98, 0.98];

%% Plot 3: Individual Model Probabilities
subplot(1, 3, 3);

% Create color array for each model using copper colormap
model_copper_colors = copper(n_models);
bar_colors = model_copper_colors(:, 1:3);  % Remove alpha channel

% Create bar plot with copper colors and alpha
h3 = bar(1:n_models, model_prob, 'FaceColor', 'flat', 'EdgeColor', 'none', 'BarWidth', 0.7, 'FaceAlpha', 0.3);
h3.CData = bar_colors;

% Styling - remove grid and use normal font
ylim([0, max(model_prob) * 1.15]);
xlabel('Model', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Model Probability', 'FontSize', 12, 'FontWeight', 'bold');
title('Individual Model Probabilities', 'FontSize', 14, 'FontWeight', 'normal', 'Color', colors.dark_gray);

% Better x-axis labels
set(gca, 'XTick', 1:n_models, 'XTickLabel', model_names, ...
    'XTickLabelRotation', 45, 'FontSize', 10, ...
    'Box', 'off', 'LineWidth', 1.2);
grid off;  % Explicitly remove grid

% Add value labels on bars
for i = 1:n_models
    if model_prob(i) > 0.01  % Only label bars with meaningful height
        text(i, model_prob(i) + max(model_prob) * 0.02, sprintf('%.3f', model_prob(i)), ...
             'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
             'FontSize', 9, 'Color', colors.dark_gray);
    end
end

ax3 = gca;
ax3.Color = [0.98, 0.98, 0.98];

%% Overall figure improvements
% sgtitle('Family-wise Bayesian Model Selection Results', ...
%     'FontSize', 16, 'FontWeight', 'normal', 'Color', colors.dark_gray);

% Adjust subplot spacing
set(fig, 'Units', 'normalized');
subplot(1,3,1); pos1 = get(gca, 'Position'); pos1(1) = 0.08; set(gca, 'Position', pos1);
subplot(1,3,2); pos2 = get(gca, 'Position'); pos2(1) = 0.38; set(gca, 'Position', pos2);
subplot(1,3,3); pos3 = get(gca, 'Position'); pos3(1) = 0.68; pos3(3) = 0.28; set(gca, 'Position', pos3);

%% 6. Save Results
results_summary = struct();
results_summary.family_names = family_names;
results_summary.families = families;
results_summary.family_probabilities = family_prob;
results_summary.family_exceedance_probabilities = family_xp;
results_summary.model_names = model_names;
results_summary.model_probabilities = model_prob;
results_summary.model_exceedance_probabilities = model_xp;

% Save to file
save('family_wise_BMS_results_perceptual.mat', 'results_summary', 'posterior', 'out');
fprintf('\nResults saved to: family_wise_BMS_results.mat\n');

%% 7. Additional Analysis: Model Contributions within Families
fprintf('\n=== WITHIN-FAMILY MODEL CONTRIBUTIONS ===\n');
for f = 1:length(families)
    fprintf('\nFamily %d (%s):\n', f, family_names{f});
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);
    
    % Normalize to get relative contributions within family
    if sum(family_model_probs) > 0
        relative_contrib = family_model_probs / sum(family_model_probs);
        for m = 1:length(models_in_family)
            model_idx = models_in_family(m);
            fprintf('  %s (Model %d): %.3f (%.1f%% of family)\n', ...
                model_names{model_idx}, model_idx, family_model_probs(m), ...
                relative_contrib(m) * 100);
        end
    end
end

% ---------- Stacked bars per-family (no hard-coded probs) ----------
num_families = length(families);

% Choose base colors for each family (customize if you want)
if num_families == 1
    family_colors = [0.2 0.6 0.9];
elseif num_families == 2
    family_colors = [
        0.0000 0.4470 0.7410;   % blue-ish
        0.9 0.3250 0.0980    % orange-ish
    ];
else
    family_colors = parula(num_families); % fallback for >2 families
end

figure;
hold on;
handles = gobjects(0);   % for legend handles
labels  = {};            % for legend labels

% Plot each family as a single stacked bar (so colors can be set per-family)
for f = 1:num_families
    models_in_family = families{f};
    family_model_probs = model_prob(models_in_family);
    if sum(family_model_probs) > 0
        rel = family_model_probs / sum(family_model_probs);  % relative contributions
    else
        rel = zeros(size(family_model_probs));
    end

    % Create shades for models within the family (darker -> lighter)
    n = numel(rel);
    % shades from 0.55 (darker) to 1.0 (base color)
    shades = linspace(0.35, 1.00, n)';  
    base = family_colors(min(f,size(family_colors,1)), :);
    model_colors = bsxfun(@times, shades, base); % n x 3 matrix

    % Draw a single stacked bar at x = f
    b = bar(f, rel, 0.6, 'stacked');   % b is an array of Bar objects (one per segment)
    for i = 1:numel(b)
        % set color for this segment (this family only)
        b(i).FaceColor = model_colors(i,:);
        b(i).EdgeColor = 'none';

        % store handle & label for legend (label with model name + family)
        model_idx = models_in_family(i);
        handles(end+1) = b(i); %#ok<SAGROW>
        labels{end+1} = sprintf('%s (Family: %s)', model_names{model_idx}, family_names{f});
    end

    % optional: add text labels inside segments (percent)
    cum = 0;
    for i = 1:n
        h = rel(i);
        if h > 0
            yc = cum + h/2;
            % choose text color for contrast
            rgb = model_colors(i,:);
            lum = 0.299*rgb(1) + 0.587*rgb(2) + 0.114*rgb(3);
            txtc = 'w';
            if lum > 0.7, txtc = 'k'; end
            text(f, yc, sprintf('%.1f%%', h*100), ...
                 'HorizontalAlignment','center', 'VerticalAlignment','middle', ...
                 'FontSize',9, 'Color', txtc);
        end
        cum = cum + h;
    end
end

% Format axes & legend
xlim([0.5, num_families+0.5]);
set(gca, 'XTick', 1:num_families, 'XTickLabel', family_names, 'FontSize', 12);
ylabel('Relative Model Contribution (within family)');
title('Within-Family Model Contributions (stacked)');
ylim([0 1]);
grid on; box on;

% Legend (one entry per model segment)
legend(handles, labels, 'Location', 'eastoutside');

hold off;



%% 8. Statistical Summary
fprintf('\n=== STATISTICAL SUMMARY ===\n');
fprintf('Winning Family: %s (Prob = %.3f, XP = %.3f)\n', ...
    family_names{find(family_prob == max(family_prob))}, ...
    max(family_prob), max(family_xp));
fprintf('Winning Model: %s (Prob = %.3f, XP = %.3f)\n', ...
    model_names{find(model_prob == max(model_prob))}, ...
    max(model_prob), max(model_xp));

% Evidence strength interpretation
if max(family_xp) > 0.95
    evidence_strength = 'Very Strong';
elseif max(family_xp) > 0.90
    evidence_strength = 'Strong';
elseif max(family_xp) > 0.75
    evidence_strength = 'Moderate';
else
    evidence_strength = 'Weak';
end
fprintf('Evidence Strength: %s (XP = %.3f)\n', evidence_strength, max(family_xp));

%% Manual Family BMS Implementation (fallback function)
function [posterior, out] = manual_family_BMS(lme, families, family_names)
    fprintf('Implementing family-wise BMS manually...\n');
    
    [n_subjects, n_models] = size(lme);
    n_families = length(families);
    
    % Step 1: Compute family-level log evidence
    family_lme = zeros(n_subjects, n_families);
    
    for f = 1:n_families
        models_in_family = families{f};
        % For each subject, compute log sum exp of models in family
        for s = 1:n_subjects
            family_evidence = lme(s, models_in_family);
            % Numerical stable log-sum-exp
            max_ev = max(family_evidence);
            family_lme(s, f) = max_ev + log(sum(exp(family_evidence - max_ev)));
        end
    end
    
    % Step 2: Run BMS on family-level evidence using SPM functions
    try
        [alpha_fam, exp_r_fam, xp_fam] = spm_BMS(family_lme, 1e6, 0, 0, 1, ones(1, n_families));
    catch
        % Fallback if SPM not available
        fprintf('SPM not available, using simplified BMS...\n');
        % Simple softmax over mean log evidence
        mean_family_lme = mean(family_lme, 1);
        exp_r_fam = exp(mean_family_lme - max(mean_family_lme));
        exp_r_fam = exp_r_fam / sum(exp_r_fam);
        xp_fam = exp_r_fam;  % Simplified exceedance probabilities
    end
    
    % Step 3: Compute model-level probabilities within families
    model_prob = zeros(1, n_models);
    for f = 1:n_families
        models_in_family = families{f};
        if length(models_in_family) == 1
            % Single model in family
            model_prob(models_in_family) = exp_r_fam(f);
        else
            % Multiple models in family - run BMS within family
            family_models_lme = lme(:, models_in_family);
            try
                [~, exp_r_within, ~] = spm_BMS(family_models_lme, 1e6, 0, 0, 1, []);
            catch
                % Fallback method
                mean_model_lme = mean(family_models_lme, 1);
                exp_r_within = exp(mean_model_lme - max(mean_model_lme));
                exp_r_within = exp_r_within / sum(exp_r_within);
            end
            % Weight by family probability
            model_prob(models_in_family) = exp_r_fam(f) * exp_r_within;
        end
    end
    
    % Step 4: Format output
    posterior.r = exp_r_fam;           % Family probabilities
    posterior.r_model = model_prob;     % Model probabilities
    
    out.families.ep = xp_fam;          % Family exceedance probabilities
    out.ep = zeros(1, n_models);       % Model exceedance probabilities
    
    % Approximate model exceedance probabilities
    for m = 1:n_models
        out.ep(m) = model_prob(m);  % Simplified approximation
    end
    
    fprintf('Manual family BMS completed.\n');
end
