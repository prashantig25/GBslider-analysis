function bar_plots_pval(y,mean_all,SEM_all,n,x_groups,bars,legend_names,xticks,xticklabs,title_name, ...
    xlabelname,ylabelname,disp_pval,scatter_dots,dot_size,varargin)   
    %

    % mean_all, SEM_all should be arranged as n x 1
    % plot
    leg_len = length(legend_names);
    x = [];
    for i = 1:x_groups
        x = [x; repelem(i,n,1)];
    end
    data_plot = [];
    for i = 1:x_groups
        data_plot = [data_plot; nanmean(y(x==i,:))];
    end
    %x = [repelem(1,n,1);repelem(2,n,1);repelem(3,n,1);repelem(4,n,1)];
    % h=bar([mean(y(x==1,:));mean(y(x==2,:));mean(y(x==3,:));mean(y(x==4,:))]);
    h = bar(data_plot);
%     h.BarWidth = 0.6;
    hold on
    if nargin > 15
    for b = 1:bars
        h(b).FaceColor = varargin{1}(b,:);
    end
    end
    hold on
%     for l = 1:leg_len
%         if l < leg_len
%             legend(legend_names{l})
%             hold on
%         else
%             legend(legend_names{l},'AutoUpdate','off')
%             hold on
%         end
%     end
    legend(legend_names,"AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none')
    
   if scatter_dots == 1
        for b = 1:bars
            for i = 1:x_groups
                scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), y(x==i,b),dot_size,"o", ...
                    'MarkerEdgeColor','k','MarkerFaceColor','auto','XJitter','randn','XJitterWidth',.5)
                varargin{3}(i) = max(y(x==i,b));
            end
        end
        if nargin > 20
            for b = 1:bars
                for i = 1:x_groups
                    y_group = y(x==i,b);
                    y_single = y_group(varargin{6});
                    single = scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), y_single,40,"o", ...
                        'MarkerEdgeColor','k','MarkerFaceColor',[20, 55, 108]./256,'XJitter','randn','XJitterWidth',.5);
                end
            end
        end
   end

    ngroups = size(mean_all, 1); % y_SEM is avg arranged as single means
    nbars = size(mean_all, 2);

    % Calculating the width for each bar group
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        a = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(a, mean_all(:,i),SEM_all(:,i),  'k', 'linestyle', 'none','LineWidth',1);
    end
    hold on
    if nargin > 18
        for b = 1:bars
            for i = 1:x_groups
                s = scatter(repmat(h(b).XEndPoints(i), sum(x==i),1), varargin{4}(1,i),40, ...
                    "diamond",'MarkerEdgeColor','k','MarkerFaceColor',[158, 188, 226]./256, ...
                    'XJitter','randn','XJitterWidth',.2);
            end
            if nargin > 20
                l = legend([h single(1) s(1)],{'Empirical data','Example participant','Normative agent'}, ...
                    "AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none');
                l.ItemTokenSize = [5 5];
            else
                l = legend([h s(1)],{'Empirical data','Normative agent'}, ...
                    "AutoUpdate","off",'Location','best','Box','off','Color','none','EdgeColor','none');
                l.ItemTokenSize = [5 5];
            end
        end
    end
    if nargin > 19
        xlim(varargin{5})
    end
    set(gca,'XTick',xticks)
    set(gca,'XTicklabels',xticklabs)
    box off
    title(title_name,'FontWeight','normal')
    xlabel(xlabelname)
    ylabel(ylabelname)
    set(gca,'LineWidth',1)
    set(gca,'fontname','arial') 
    pl = gca;
    pl.FontSize = 8;
    for b=1:bars
        for i = 1:x_groups
            max_y = max(y(x==i,b));
        end
    end
    % add bar labels

    xt = get(gca, 'XTick');

    if disp_pval == 1
        labels = varargin{2};
        text(xt, varargin{3},labels, 'HorizontalAlignment','center', ['' ...
            'VerticalAlignment'],'bottom','FontSize',10,'FontWeight','bold')
    end
%     axis tight