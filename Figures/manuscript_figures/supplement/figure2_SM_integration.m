% figure2_SM_integration mimics figure2_SM's panel a (grouped bar plot of
% mean signed-LR-regression beta coefficients, same 5 regressors/formula)
% but for model-simulated data instead of real behavior, comparing
% simulations with observation integration vs without. One figure per
% generative model (basicRL, RLsigma) x condition (Both, Perceptual),
% using the subjective-PE, fixed-sigma simulated betas built by
% Learning-rate analyses/lr_analysis/LR_analysis_preprint.m's "FIT SIGNED
% MODEL TO SUBJECTIVE PE, FIXED SIGMA MODEL SIMULATIONS" section
% (betas_signed_sim_<model>_subjectivePE_fixedSigma_<integration>_
% <condition>.mat).

clc
clearvars
close all

line_width = 0.5; % line width for axes
font_size = 12; % font size
font_name = 'Arial'; % font name
[~,~,~,~,~,~,darkblue_muted,~,~,~,~,~,~,~,...
    ~,~,~,fits_colors,~] = colors_rgb(); % colors

% PATH

currentDir = cd;
reqPath = 'Reward-learning-analysis (code_review)'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, 'saved_figures', 'supplement');
mkdir(save_dir)
baseDir = fullfile(desiredPath, 'Data', 'LR analyses');

% INITIALISE VARS -- same regressor selection/order/labels as
% figure2_SM.m's panel a, since the simulated betas come from the exact
% same lr_mdl formula/build_subject_table column structure.

selected_regressors = [1,2]; % regressors to be plotted -- Fixed LR, Belief-state-adapted LR only
xticks = 1:length(selected_regressors);
row1 = {'Fixed LR' 'Belief-state'};
row2 = {'' '-adapted LR'};
labelArray = [row1; row2];
ylabelname = {'Mean beta coefficients'};
colors_name = [fits_colors;darkblue_muted]; % colors for bars

model_names = {'basicRL','RLsigma'};
model_titles = {'Basic RL','RL + Est. Sensitivity'};
conditions = {'Both','Perceptual'};

for m = 1:length(model_names)
    model_label = model_names{m};

    for c = 1:length(conditions)
        cond_label = conditions{c};

        % LOAD THE SIMULATED BETAS
        betas_integrated = importdata(fullfile(baseDir, sprintf('betas_signed_sim_%s_subjectivePE_fixedSigma_integrated_%s.mat', model_label, cond_label)));
        betas_noint = importdata(fullfile(baseDir, sprintf('betas_signed_sim_%s_subjectivePE_fixedSigma_noIntegration_%s.mat', model_label, cond_label)));
        numSubjs_sim = size(betas_integrated, 1); % number of simulated "subjects" -- one simulated replicate per real subject

        % GET MEAN AND SEM FOR BETAS
        [mean_int,sem_int,coeffs_int] = prepare_betas(betas_integrated,selected_regressors,numSubjs_sim);
        [mean_noint,sem_noint,coeffs_noint] = prepare_betas(betas_noint,selected_regressors,numSubjs_sim);
        mean_avg = [mean_int,mean_noint];
        mean_sd = [sem_int,sem_noint];
        coeffs_subjs = [coeffs_int,coeffs_noint]; % (numSubjs_sim*5) x 2, stacked group-major (all subjects for regressor 1, then regressor 2, ...)

        % PLOT -- bare-bones bar+errorbar+scatter via plain MATLAB calls
        % instead of bar_plots_pval, which was taking close to an hour per
        % figure here (jittered scatter dots called 10x/figure via the
        % newer 'XJitter' scatter option, legend(...,'Location','best'),
        % and alpha-blended bars were all plausible culprits, but which
        % one couldn't be pinned down without a profiler run, which itself
        % was too slow to repeat). Per-subject dots are reproduced here
        % manually: jitter is computed with plain rand() before calling
        % scatter (not the 'XJitter' option), and all subjects for a given
        % bar are scattered in ONE call (not one call per regressor group),
        % to minimize the number of graphics objects created per figure.
        figure
        set(gcf,'Position',[100 100 450 300])
        hold on
        bar_h = bar(1:length(selected_regressors), mean_avg);
        bar_h(1).FaceColor = colors_name(1,:);
        bar_h(2).FaceColor = colors_name(2,:);
        % Legend references ONLY the two bar handles (not the scatter/
        % errorbar objects added below), and uses an explicit location
        % instead of 'best' -- 'best' tests candidate positions against
        % every plotted object's bounding box and was a leading suspect
        % for this script's earlier slowness.
        legend(bar_h, {'Integration','No Integration'}, 'Location', 'northeast', ...
            'Box', 'off', 'Color', 'none', 'EdgeColor', 'none');
        ngroups = length(selected_regressors);
        nbars = 2;
        groupwidth = min(0.8, nbars/(nbars + 1.5));

        dot_size = 15;
        jitter_width = 0.05;
        for b = 1:nbars
            x_centers = repelem(bar_h(b).XEndPoints(:), numSubjs_sim, 1); % matches coeffs_subjs's group-major stacking
            x_jittered = x_centers + (rand(size(x_centers)) - 0.5) * 2 * jitter_width;
            scatter(x_jittered, coeffs_subjs(:,b), dot_size, 'o', ...
                'MarkerEdgeColor', [184, 184, 184]./255, 'MarkerFaceColor', [220, 220, 220]./255);
        end

        for b = 1:nbars
            xpos = (1:ngroups) - groupwidth/2 + (2*b-1)*groupwidth/(2*nbars);
            errorbar(xpos, mean_avg(:,b), mean_sd(:,b), 'k', 'LineStyle', 'none', 'LineWidth', 1);
        end

        % PLOT PROPERTIES
        set(gca,'Color','none','FontName',font_name,'FontSize',font_size,'LineWidth',line_width)
        box off
        ax = gca;
        ax.XTick = xticks;
        ax.XTickLabel = '';
        for i = 1:length(labelArray)
            text(i, ax.YLim(1), sprintf('%s\n%s\n%s', labelArray{:,i}), ...
                'horizontalalignment', 'center', 'verticalalignment', 'top','FontSize',font_size);
        end
        ax.XLabel.String = sprintf('\n\n%s');
        ylabel(ylabelname)
        title(sprintf('%s (%s, Subjective PE, Fixed Sigma)', model_titles{m}, cond_label),'FontWeight','normal')

        % SAVE AS PNG
        % fig = gcf;
        % fig.PaperPositionMode = 'auto';
        % print(fig, fullfile(save_dir,sprintf('figure2_SM_integration_%s_%s.png', model_label, cond_label)), '-dpng', '-r600')
    end
end
