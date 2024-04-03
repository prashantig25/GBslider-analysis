function h = bar_plots_pval(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,disp_pval,scatter_dots,dot_size,plot_err,fontsize,linewidth,fontname,disp_legend,varargin)  
    
    % function bar_plots_pval creates bar plots with multiple bars, single data scatter
    % points, SEM bars, and displays significance stars.
    %
    % INPUTS:
    %   y: array with single data points with size n x b where n is
    %   number of data points and b is number of bars.
    %   mean_all: mean across single data points with size n x b where n
    %   is number of groups on x-axis and b is the number of bars.
    %   SEM_all: SEM across single data points with size n x b where n
    %   is number of groups on x-axis and b is the number of bars.
    %   n: number of single data points for each group on x-axis.
    %   x_groups: number of groups to be plotted on x-axis.
    %   bars: number of bars.
    %   legend_names: cell array containing legend for bars.
    %   xticks: array of tick numbers for x-axis
    %   xticklabs: cell array with tick label for x-axis
    %   title_name: cell array containing name of the plot
    %   xlabelname: cell array containing label for x-axis
    %   ylabelname: cell array containing label for y-axis
    %   disp_pval: if significance stars should be displayed
    %   scatter_dots: if single data scatter points to be plotted
    %   dot_size: specify size of single data scatter points
    %   plot_err: if errorbar needs to be plotted
    %   fontsize: font size for text
    %   linewidth: line width for plot
    %   fontname: font
    %   disp_legend: whether legend should be displayed
    %   varargin{1}: face color for bars
    %   varargin{2}: cell array containing bar labels
    %   varargin{3}: array with max value for each bar, to be used as y-axis 
    %   location for significance stars
    %   varargin{4}: means of any other dataset
    %   varargin{5}: adjusted xlims values
    %   varargin{6}: example participant to be highlighted
    %   varargin{7}: whether to use max values or user defined
    
	% ARRAY CONTAINING X-AXIS GROUP NUMBER FOR EACH DATA POINT
    x = [];
    for i = 1:x_groups
        x = [x; repelem(i,n,1)];
    end

    % CALCULATING MEAN OF Y FOR EACH GROUP ON X-AXIS
    data_plot = [];
    for i = 1:x_groups
        data_plot = [data_plot; nanmean(y(x==i,:))];
    end

    % PLOT BARS
    h = bar(1:x_groups,data_plot);
    hold on

    % CUSTOMIZE BAR APPEARANCE
    if nargin > 19
      for b = 1:bars
          h(b).FaceColor = varargin{1}(b,:);
          h(b).EdgeColor = varargin{1}(b,:);
          h(b).LineWidth = 1;
          h(b).FaceAlpha = 0.5;
      end
    end
    hold on

    % UPDATE LEGEND
    if disp_legend == 1
        l = legend(legend_names,"AutoUpdate","off",'Location','best','Box','off','Color', ...
            'none','EdgeColor','none');
        l.ItemTokenSize = [7,7];
    end

  	% PLOT SINGLE DATA POINTS ON THE BARS, FOR EACH BAR, ACROSS GROUPS ON
    % X-AXIS
   if scatter_dots == 1
        for b = 1:bars
            for i = 1:x_groups
                scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), y(x==i,b),dot_size,"o", ...
                    'MarkerEdgeColor',[184, 184, 184]./255,'MarkerFaceColor',[220, 220, 220]./255,'XJitter', ...
                    'randn','XJitterWidth',0.2,'YJitterWidth',0.2)
%                 varargin{3}(i) = max(y(x==i,b));
            end
        end
        if nargin > 25 % EXAMPLE PARTICIPANT SCATTER POINT TO BE HIGHLIGHTED
            for b = 1:bars
                for i = 1:x_groups
                    y_group = y(x==i,b);
                    y_single = y_group(varargin{6});
                    single = scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), y_single,dot_size*2,"o", ...
                        'MarkerEdgeColor','k','MarkerFaceColor',[20, 55, 108]./256,'XJitter','randn','XJitterWidth',.5);
                end
            end
        end
   end


	% CALCULATING THE WIDTH FOR EACH GROUP ON X-AXIS
    if plot_err == 1
      ngroups = size(mean_all, 1); % y_SEM is avg arranged as single means
      nbars = size(mean_all, 2);
      groupwidth = min(0.8, nbars/(nbars + 1.5));
      for i = 1:nbars
          a = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
          errorbar(a, mean_all(:,i),SEM_all(:,i),  'k', 'linestyle', 'none','LineWidth',1);
      end
      hold on
    end

    % ADD MEANS OF ANY OTHER DATASET (e.g., normative agent)
    if nargin > 23
        for b = 1:bars
            for i = 1:x_groups
                s = scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), varargin{4}(1,i),dot_size*4, ...
                    "diamond",'MarkerEdgeColor','k','MarkerFaceColor',[158, 188, 226]./256, ...
                    'XJitter','randn','XJitterWidth',.2);
            end
            if nargin > 25 && disp_legend == 1
                l = legend([h single(1) s(1)],legend_names, ...
                    "AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none');
                l.ItemTokenSize = [5 5];
            elseif disp_legend == 1
                l = legend([h s(1)],legend_names, ...
                    "AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none');
                l.ItemTokenSize = [5 5];
            end
        end
    end

    % ADJUST XLIMS
    if nargin > 23
        xlim(varargin{5})
    end

    % CUSTOMIZE OTHER PLOT PROPERTIES
    set(gca,'XTick',xticks)
    set(gca,'XTicklabels',xticklabs)
    box off
    pl = gca;
    pl.FontSize = fontsize;  
    set(gca,'LineWidth',linewidth)
    set(gca,'fontname',fontname) 

    title(title_name,'FontWeight','normal','Interpreter','tex')
    xlabel(xlabelname)
    ylabel(ylabelname)

    % ADD BAR LABELS
    xt = get(gca, 'XTick');
    if disp_pval == 1
        labels = varargin{2};
        text(xt, varargin{3},labels, 'HorizontalAlignment','center', ['' ...
            'VerticalAlignment'],'bottom','FontSize',6,'FontWeight','normal')
    end
end