clc
clearvars
colors_manuscript; % colors for the plot
line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
linewidth_axes = 0.5; % line width for axes
fontsize_label = 12; % font size for subplot labels

% INITIALISE VARS
data_subjs = readtable("data_recoding_signed.xlsx");
num_subjs = 99; % number of subjects
%%

% SET VARIABLES TO RUN THE FUNCTION
id_subjs = unique(data_subjs.run_id); % list of subject-IDs
model3 = 'up ~ pe + pe:contrast_diff + pe:salience +pe:congruence + pe:pe_sign'; % model
mdl = model3; % which regression model
pred_vars = {'pe','salience','contrast_diff','congruence','condition','reward_unc','subj_est_unc' ...
        ,'reward','mu','pe_sign','pu'}; % cell array with names of predictor variables
resp_var = 'up'; % name of response variable
cat_vars = {'salience','congruence','condition','reward_unc','pe_sign'}; % cell array with names of categorical variables
num_vars = 5; % number of predictor vars
res_subjs = []; % empty array to store residuals
weight_y_n = 0; % non-weighted regression
subjs = [61,39,48]; % example participants
%% INITIALIZE TILE
figure
set(gcf,'Position',[100 100 600 400])
t = tiledlayout(2,3);
t.TileSpacing = 'compact';
t.Padding = 'compact';

%% PLOT

axes_list = [];
for i = 1:3
    
    % FIT MODEL
    data = data_subjs(data_subjs.run_id == id_subjs(subjs(i)),:);
    tbl = table(data.pe_actual_corr,data.up_actual_corr, round(data.norm_condiff), data.contrast,...
            data.condition,data.congruence,data.reward_unc,data.norm_subjest,data.pe_sign,...
            'VariableNames',{'pe','up','contrast_diff','salience','condition','congruence' ...
            ,'reward_unc','subj_est_unc','pe_sign'});
    [betas,rsquared,residuals,coeffs_name,lm] = linear_fit(tbl,mdl,pred_vars,resp_var, ...
    cat_vars,num_vars,weight_y_n);
    
    % PLOT ADDED VARIABLE PLOT
    axes = nexttile(i,[1,1]);
    p = plotAdded(lm,[2,3,4,5,6],'Marker','o','MarkerSize',3,'MarkerEdgeColor', ...
        dots_edges,'MarkerFaceColor',gray_dots);
    p(1).Color = barface_green;
    p(2).Color = barface_green;
    p(3).Color = barface_green;
    p(2).LineWidth = 1.5;

    legend('off')
    xline(0,"LineWidth",0.2,LineStyle="--")
    yline(0,"LineWidth",0.2,LineStyle="--")
    xlabel('Adjusted regressors')
    ylabel('Adjusted updates')
    set(gca,'Color','none','FontName',font_name,'FontSize',font_size)
    set(gca,'LineWidth',linewidth_axes)
    title(strcat('Participant',{' '},num2str(subjs(i))),'FontWeight','normal')
    box off
    axes_list = [axes_list, axes];
end

for i = 1:3
    % GET EMPIRICAL and POSTERIOR UPDATES
    [betas_all,rsquared_full,residuals_reg,coeffs_name,h_vals,p_vals,t_vals,posterior_up_subjs] = get_coeffs(data_subjs, ...
        mdl,pred_vars,resp_var,cat_vars,num_vars,res_subjs,subjs(i),weight_y_n);
    y = data_subjs.up_actual_corr(data_subjs.pe_actual_corr ~= 0 & data_subjs.run_id == id_subjs(subjs(i)));
    y_hat = posterior_up_subjs;

    % PLOT
    axes = nexttile(i+3,[1,1]);
    hold on
    h1 = histfit(y_hat);
    hold on
    h = histfit(y);

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
    title('Updates','FontWeight','normal')
    axes_list = [axes_list, axes];
end

l = legend('Regression fits','','Empirical updates','','EdgeColor','none','Color','none','Location','Best');
l.ItemTokenSize = [7 7];
%% ADD SUBPLOT LABELS

ax1_pos = axes.Position;
adjust_x = -0.065;
adjust_y = ax1_pos(4);
num_labels = 6;
label_string = {'a','b','c','d','e','f'};
for n = 1:num_labels
    [label_x,label_y] = change_plotlabel(axes_list(1,n),adjust_x,adjust_y);
    annotation("textbox",[label_x label_y .05 .05],'String', ...
        label_string(n),'FontSize',fontsize_label,'LineStyle','none','HorizontalAlignment','center')
end
%%
fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, 'figure3_SM3', '-dpng', '-r600') 