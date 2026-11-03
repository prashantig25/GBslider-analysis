% figure5_SM creates figure S5 and plots regression model diagnostics for
% example participants

clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
linewidth_axes = 0.5; % line width for axes
fontsize_label = 12; % font size for subplot labels
darkblue_muted = Colors.darkblue_muted;
gray_dots = Colors.gray_dots;
barface_green = Colors.barface_green;
dots_edges = Colors.dots_edges;
fits_colors = Colors.fits_colors;

% PATH 

currentDir = cd;
reqPath = 'GBslider-analysis'; % to which directory one must save in
pathParts = strsplit(currentDir, filesep);
if strcmp(pathParts{end}, reqPath)
    disp('Current directory is already the desired path. No need to run createSavePaths.');
    desiredPath = currentDir;
else
    % Call the function to create the desired path
    desiredPath = createSavePaths(currentDir, reqPath);
end
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"supplement");
mkdir(save_dir)

% INITIALISE VARS

data_subjs = importdata(strcat(desiredPath, filesep, "Data", filesep, "LR analyses", filesep, "preprocessed_data.mat")); % single-trial updates, prediction errors
data_subjs.id = data_subjs.ID;

% SET VARIABLES TO RUN THE FUNCTION

id_subjs = unique(data_subjs.id); % list of subject-IDs
num_subjs = length(id_subjs); % number of subjects, extracted from the data instead of hardcoded
model3 = 'up ~ pe + pe:contrast_diff + pe:salience +pe:congruence + pe:pe_sign'; % model
mdl = model3; % which regression model
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','pe_sign'}; % cell array with names of categorical variables
num_vars = 5; % number of predictor vars
weight_y_n = 0; % non-weighted regression
subjs = [15,23,40]; % example participants

% INITIALIZE TILE

figure
set(gcf,'Position',[100 100 600 400])
t = tiledlayout(2,3);
t.TileSpacing = 'compact';
t.Padding = 'compact';

% PLOT

axes_list = [];
lm_subjs = cell(1,3); % store each participant's fitted model and table to reuse below instead of fitting again
tbl_subjs = cell(1,3);
for i = 1:3

    % FIT MODEL -- excludes pe == 0 trials and rounds norm_condiff to 2
    % decimal places, matching the convention used elsewhere (figure4.m,
    % figure1_SM.m) for this same model
    data = data_subjs(data_subjs.id == id_subjs(subjs(i)) & data_subjs.pe ~= 0,:);
    tbl = table(data.pe,data.up, round(data.norm_condiff,2), data.contrast,...
            data.choice_cond,data.congruence,data.reward_unc,data.pe_sign,...
            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
            ,'reward_unc','pe_sign'});
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
    cat_vars,num_vars,weight_y_n);
    lm_subjs{i} = lm;
    tbl_subjs{i} = tbl;

    % PLOT ADDED VARIABLE PLOT
    axes = nexttile(i,[1,1]);
    p = plotAdded(lm,[2,3,4,5,6],'Marker','o','MarkerSize',3,'MarkerEdgeColor', ...
        dots_edges,'MarkerFaceColor',gray_dots);

    % PLOT PROPERTIES
    p(1).Color = darkblue_muted;
    p(2).Color = darkblue_muted;
    p(3).Color = darkblue_muted;

    p(2).LineWidth = 1.5;
    p(3).LineWidth = 0.2;
    p(3).LineStyle = '--';

    legend('off')
    xline(0,"LineWidth",0.5,LineStyle="--")
    yline(0,"LineWidth",0.5,LineStyle="--")
    xlabel('Model regressor value')
    ylabel('Predicted update')
    set(gca,'Color','none','FontName',font_name,'FontSize',font_size)
    set(gca,'LineWidth',linewidth_axes)
    title(strcat('Participant',{' '},num2str(subjs(i))),'FontWeight','normal')
    box off
    xlim([-1.5,1.5])
    axes_list = [axes_list, axes];
end

for i = 1:3

    % GET EMPIRICAL and POSTERIOR UPDATES -- reuses the model already fit
    % in the loop above instead of fitting it again via get_coeffs
    y = data_subjs.up(data_subjs.pe ~= 0 & data_subjs.id == id_subjs(subjs(i)));
    y_hat = predict(lm_subjs{i},tbl_subjs{i});

    % PLOT
    axes = nexttile(i+3,[1,1]);
    hold on
    h1 = histfit(y_hat);
    hold on
    h = histfit(y);

    % PLOT PROPERTIES
    h1(1).FaceAlpha = 1; % face alpha
    h(1).FaceAlpha = 0.7;
    
    h1(2).Color = [37, 50, 55]/255; % distribution color
    h(2).Color = fits_colors;
    
    h1(1).FaceColor = [37, 50, 55]/255; % bar face color
    h(1).FaceColor = fits_colors;

    h(1).EdgeColor = fits_colors; % edge color
    h1(1).EdgeColor = [37, 50, 55]/255;

    set(gca,'Color','none','FontName',font_name,'FontSize',font_size)
    set(gca,'LineWidth',linewidth_axes)
    xlabel('Update','FontWeight','normal')
    ylabel('Frequency')
    axes_list = [axes_list, axes];
end

l = legend('Regression fits','','Empirical updates','','EdgeColor','none','Color','none','Location','Best');
l.ItemTokenSize = [7 7];

% ADD SUBPLOT LABELS

ax1_pos = axes.Position;
adjust_x = -0.065; % adjusted x-position for subplot label
adjust_y = ax1_pos(4); % adjusted y-position for subplot label
num_labels = 6; % number of subplot labels
label_string = {'a','b','c','d','e','f'};
for n = 1:num_labels
    [label_x,label_y] = change_plotlabel(axes_list(1,n),adjust_x,adjust_y);
    annotation("textbox",[label_x label_y .05 .05],'String', ...
        label_string(n),'FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
end

% SAVE AS PNG

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,filesep,'figure5_SM5'), '-dpng', '-r600') 