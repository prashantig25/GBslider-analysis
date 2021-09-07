function task_model_sims
    % This function runs simulations for the task model. 
    % The task model simulates a trial's state and state-dependent contrast
    % level. For purposes of simulation, the task model also generates an
    % action like a random agent & the model simulates an action-dependent
    % reward. 
    
    task = Task(); % import task class
    T = task.T; % number of trials
    B = task.B; % number of blocks
    
    % pre-allocation
    s = zeros(1,T);
    c = zeros(1,T);
    a = zeros(1,T);
    r = zeros(1,T);
    
    for j = 1:B % number of blocks
        for i = 1:T % number of trials in a block
            task.state_sample();
            task.contrast_sample();
            task.random_action();
            task.reward_sample();
            s(i) = task.s_t;
            c(i) = task.c_t;
            a(i) = task.a_t;
            r(i) = task.r_t;
        end
        
        % create plots
        x = 1:T;
        figure
        subplot(2,2,1)
        scatter(x,s,40,'k','filled')
        xlabel('Trials')
        ylabel('State')
        box on
        hold on
        subplot(2,2,2)
        bar(x,c)
        xlabel('Trials')
        ylabel('Contrast difference')
        box on
        hold on
        subplot(2,2,3)
        scatter(x,a,40,'k','filled')
        xlabel('Trials')
        ylabel('Random action')
        box on
        hold on
        subplot(2,2,4)
        scatter(x,r,40,'k','filled')
        box on
        xlabel('Trials')
        ylabel('Reward')
        ylim([0,1]);
        filename = ['task_simulations_2_', num2str(j)];
        sgt = sgtitle('High Perceptual & Low Reward Uncertainty')
        sgt.FontSize = 10;
        saveas(gcf,filename,'png') 
        saveas(gcf,filename,'fig')
    end
end
