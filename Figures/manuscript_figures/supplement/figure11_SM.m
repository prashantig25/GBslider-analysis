clc
clearvars

line_width = 0.5; % line width for axes
font_size = 6; % font size
font_name = 'Arial'; % font name
linewidth_plot = 1; % line width for plot lines
[~,~,~,~,~,~,darkblue_muted,mix,perc,rew,~,~,~,~,...
   ~,~,~,~,~] = colors_rgb(); % colors
trials = 25; % number of trials
num_simulations = 396; % number of simulations
num_plotted = 100; % number of simulations to be plotted

% PATH

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
save_dir = fullfile(desiredPath, filesep, "saved_figures",filesep,"supplement");
mkdir(save_dir)

agent_mix = readtable(fullfile(desiredPath, "Data", "agent simulations", "data_agent_condition1.txt"));
agent_perc = readtable(fullfile(desiredPath, "Data", "agent simulations", "data_agent_condition2.txt"));
agent_rew = readtable(fullfile(desiredPath, "Data", "agent simulations", "data_agent_condition3.txt"));
agent_mix_highPU = readtable(fullfile(desiredPath, "Data", "agent simulations", "data_agent_condition1_highPU.txt"));

% ADD SIMULATION NUMBER
simulation_number = [];
for n = 1:num_simulations
    simulation_number = [simulation_number;repelem(n,trials,1)];
end
agent_mix.simulation = simulation_number;
agent_perc.simulation = simulation_number;
agent_rew.simulation = simulation_number;
agent_mix_highPU.simulation = simulation_number;

% GET PERFORMANCE and SLIDER
perf_mix = NaN(num_plotted,1);
perf_perc = NaN(num_plotted,1);
perf_rew = NaN(num_plotted,1);

mu_mix = NaN(num_plotted,trials);
mu_perc = NaN(num_plotted,trials);
mu_rew = NaN(num_plotted,trials);
mu_mix_highPU = NaN(num_plotted,trials);
for n = 1:num_plotted
    perf_mix(n,1) = nanmean(agent_mix.correct(agent_mix.simulation ==n),1);
    perf_perc(n,1) = nanmean(agent_perc.correct(agent_perc.simulation ==n),1);
    perf_rew(n,1) = nanmean(agent_rew.correct(agent_rew.simulation ==n),1);

    mu_mix(n,:) = agent_mix.mu(agent_mix.simulation ==n);
    mu_perc(n,:) = agent_perc.mu(agent_perc.simulation ==n);
    mu_rew(n,:) = agent_rew.mu(agent_rew.simulation ==n);
    mu_mix_highPU(n,:) = agent_mix_highPU.mu(agent_mix_highPU.simulation ==n);
end
perf = [nanmean(perf_mix);nanmean(perf_perc);nanmean(perf_rew)];
mu_mix = nanmean(mu_mix);
mu_perc = nanmean(mu_perc);
mu_rew = nanmean(mu_rew);
mu_mix_highPU = nanmean(mu_mix_highPU);
%% INITIALISE TILE LAYOUT

figure
set(gcf,'Position',[100 100 350 175])
t = tiledlayout(1,2);
t.TileSpacing = 'compact';
% t.Padding = 'compact';
%% PLOT PERFORMANCE and SLIDER

ax1 = nexttile(1,[1,1]);
hold on
b = bar(perf,'FaceColor','flat','EdgeColor','flat','FaceAlpha',0.5,'LineWidth',1,'BarWidth',0.5);
b(1).CData = darkblue_muted;
hold on
ylim([0.5,1])
set(gca,'Color','none','FontName','Arial','FontSize',6)
xticklabels(["Both","Perceptual","Reward"])
xticks([1,2,3])
xlim([0.4,3.6])
ylabel('P(Correct)')

ax2 = nexttile(2,[1,1]);
hold on
plot(1:trials,mu_mix,'LineStyle','-','Color',mix,'LineWidth',2)
hold on
plot(1:trials,mu_perc,'LineStyle','-','Color',perc,'LineWidth',2)
hold on
plot(1:trials,mu_rew,'LineStyle','-','Color',rew,'LineWidth',2)
hold on
plot(1:trials,mu_mix_highPU,'LineStyle','-','Color',[172, 207, 230]./255,'LineWidth',2)

xlim([1,25])
yline(0.7,'LineStyle','--','LineWidth',0.5)
yline(0.9,'LineStyle','--','LineWidth',0.5)
set(gca,'Color','none','FontName','Arial','FontSize',6)
xlabel('Trial')
ylabel({'Learned' 'contingency parameter'})
l = legend('Both','Perceptual','Reward','High uncertainty','Actual \mu','EdgeColor','none', ...
    'Color','none','Location','Best','Interpreter','Tex','FontSize',5.5,'NumColumns',2);
l.ItemTokenSize = [7 7];
ylim([0.5,1])
%% ADD SUBPLOT LABELS

ax1_pos = ax1.Position;
adjust_x = [-0.08,-0.08]; % adjusted x-position for subplot label
adjust_y = ax1_pos(4) + 0.05; % adjusted y-position for subplot label
[label_x,label_y] = change_plotlabel(ax1,adjust_x(1),adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'a','FontSize',12,'LineStyle','none','HorizontalAlignment','center')

[label_x,label_y] = change_plotlabel(ax2,adjust_x(2),adjust_y);
annotation("textbox",[label_x label_y .05 .05],'String', ...
    'b','FontSize',12,'LineStyle','none','HorizontalAlignment','center')
%% SAVE FIGURE

fig = gcf; % use `fig = gcf` ("Get Current Figure") if want to print the currently displayed figure
fig.PaperPositionMode = 'auto'; % To make Matlab respect the size of the plot on screen
print(fig, fullfile(save_dir,'agent_simulations1.png'), '-dpng', '-r600') 