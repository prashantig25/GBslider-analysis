function axes_new = plot_splithalf_corr(betas_split0,betas_split1,vars,axes_old,line_color,title_variables,font_name,font_size,line_width)
    % function plot_splithalf_corr plots odd-vs-even split-half
    % reliability scatter plots (fitted line + confidence interval, and
    % an r/p-value title) for each regressor in vars, replacing each
    % axes in axes_old with a repositioned new axes.
    %
    % INPUTS:
    %   betas_split0: betas for the odd-trial split
    %   betas_split1: betas for the even-trial split
    %   vars: column indices of the regressors to plot
    %   axes_old: array of existing axes handles to replace, one per
    %   regressor in vars
    %   line_color: color for the fitted line and confidence interval
    %   title_variables: string array with a regressor-name header to
    %   prepend to each title, or [] to omit the header (deliberately
    %   different between callers -- see conversation)
    %   font_name: font name
    %   font_size: font size
    %   line_width: line width for axes
    %
    % OUTPUTS:
    %   axes_new: array of the newly created axes handles

    axes_new = axes_old;
    for i = 1:length(axes_old)

        % GET CORRELATION AND FORMAT P-VALUE
        [rho,pval] = corr(betas_split0(:,vars(i)),betas_split1(:,vars(i)));
        if pval < 0.001
            pval_str = '\itp\rm < 0.001';
        else
            pval_str = strcat("\itp\rm = ",num2str(round(pval,3)));
        end

        % CHANGE AXES POSITION
        new_pos = change_position(axes_old(i),[0 0 0 -0.05]); % new position for axis
        axes_new(i) = axes('Units', 'Normalized', 'Position', new_pos); % update the position
        delete(axes_old(i)); % delete old axes

        % PLOT
        hold on
        scatter(betas_split0(:,vars(i)),betas_split1(:,vars(i)),'filled','o', ...
            'MarkerFaceColor',[220, 220, 220]./255, ...
            'MarkerEdgeColor',[184, 184, 184]./255,'LineWidth',0.7,'MarkerFaceAlpha',1,SizeData=10);
        hold on

        % PLOT PROPERTIES
        xlabel('Odd trials')
        ylabel('Even trials')
        title_str = strcat("\itr\rm = ",{''},num2str(round(rho,2))) + newline + pval_str;
        if ~isempty(title_variables)
            title_str = title_variables(i) + newline + title_str;
        end
        title(title_str,'FontWeight','normal')
        adjust_figprops(axes_new(i),font_name,font_size,line_width)
        box("off")

        % CONFIDENCE INTERVALS
        mdl = fitlm(betas_split0(:,vars(i)),betas_split1(:,vars(i))); % fit a linear model
        xrange = xlim;
        xvalues = linspace(xrange(1), xrange(2), 100); % predictor values
        [yfit, yci] = predict(mdl, xvalues'); % compute and plot confidence interval lines
        hold on
        plot(xvalues, yfit,LineWidth=1.5, Color=line_color); % plot the fitted line
        plot(xvalues, yci, Color=line_color, LineWidth=0.2,LineStyle='--'); % plot the upper confidence interval
        plot(xvalues, yci(:,1), Color=line_color, LineWidth=0.2,LineStyle='--'); % plot the lower confidence interval
    end
end
