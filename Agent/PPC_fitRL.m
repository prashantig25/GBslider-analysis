clc
clearvars

alphaParameter = importdata("alphaParameterPerceptual.mat");
betaParameter = importdata("betaParameterPerceptual.mat");

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);
dataContrast = data(data.contrast ==0,:);
dataContrast = dataContrast(dataContrast.choice_cond == 2,:);
ecoperfPosterior = NaN(numSubjs,1);
ecoperfActual = NaN(numSubjs,1);

q_0_0 = 0.5;

for i = 1:length(uniqueID)

    dataSubj = dataContrast(dataContrast.ID == uniqueID(i),:);
    ecoperfActual(i) = nanmean(dataSubj.ecoperf);

    % Prepare your data
    rewards = dataSubj.recoded_reward;%(dataSubj.blocks == 1); % Vector of outcomes
    state = dataSubj.state;%(dataSubj.blocks == 1);
    condiff = dataSubj.con_diff_choice;

    n_trials = height(dataSubj);

    for t = 1:n_trials

        if mod(t,25) == 0
            q_0_0 = 0.5;
        end

        % Compute choice probabilities using softmax
        q_0_1 = 1-q_0_0;
        q_1_0 = 1-q_0_0;
        q_1_1 = q_0_0;  
        Qvalues = [q_0_0, q_0_1];

        p_a_t = exp(Qvalues.*betaParameter(i)) / sum(exp(Qvalues.*betaParameter(i)));

        % Avoid log(0) by setting lower bounds
%         p1 = max(p_a_t(1), 1e-10);
%         p2 = max(p_a_t(2), 1e-10);

        [~, argmax] = max(p_a_t);
        choices(t) = argmax - 1;

        if state(t) == 1
            choices(t) = 1-choices(t);
        end
        
        % Generate reward
        if state(t) == 0
            if choices(t) == 0
                rewards(t) = (rand() < 0.9);  % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.1);  % 30% chance of reward for incorrect choice
            end
        else
            if choices(t) == 1
                rewards(t) = (rand() < 0.9);  % 70% chance of reward for correct choice
            else
                rewards(t) = (rand() < 0.1);  % 30% chance of reward for incorrect choice
            end
        end

        if choices(t) == 1
            rewards(t) = 1-rewards(t);
        end
              
        % Update Q-values based on reward
        if state(t) == 0
            q_0_0 = q_0_0 + alphaParameter(i) .* (rewards(t) - q_0_0);
        else
            q_0_0 = q_0_0 + alphaParameter(i) .* ((1-rewards(t)) - q_0_0);
        end
    end

    posteriorTable = table(choices.', state, rewards, repelem(i,n_trials).','VariableNames',{'action','state','rewards','ID'});

    for t = 1:n_trials
        if posteriorTable.state(t) == posteriorTable.action(t)
            posteriorTable.ecoperf(t) = 1;
        else
            posteriorTable.ecoperf(t) = 0;
        end
    end
    ecoperfPosterior(i) = nanmean(posteriorTable.ecoperf);
end

figure
hold on
scatter(ecoperfActual,ecoperfPosterior)
lsline
xlabel("Actual economic performance")
ylabel("Posterior economic performance")
title("Perceptual condition")