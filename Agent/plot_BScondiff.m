clc
clearvars

data = importdata("preprocessed_dataFitting.mat");
uniqueID = unique(data.ID);
numSubjs = length(uniqueID);
sigma = importdata("sigmaNoReward_simpleRL.mat");

dataContrast = data;
% dataContrast = data(data.contrast ==1,:);
dataContrast = dataContrast(dataContrast.choice_cond ~= 3,:);
dataContrast.condiff_relative = (dataContrast.contrast_left - dataContrast.contrast_right)./2;
nbins = 10;  % Number of bins
binEdges = prctile(dataContrast.con_diff_choice, linspace(0, 100, nbins + 1));  % Percentile-based edges
dataContrast.bins = discretize(dataContrast.con_diff_choice, binEdges);
BS_binned = NaN(numSubjs,nbins);

for n = 1:numSubjs
    dataSubj = dataContrast(dataContrast.ID == uniqueID(n),:);

    belief_state = NaN(height(dataSubj),1);
    % Compute belief about perceptual state based on σ
    for h = 1:height(dataSubj)
        if dataSubj.state(h) == 1
            belief_state(h,1) = normcdf(dataSubj.condiff_relative(h), 0, sigma(n));
        else
            belief_state(h,1) = 1-normcdf(dataSubj.condiff_relative(h), 0, sigma(n));
        end
        belief_state(h,1) = belief_state(h,1)-0.5;
    end

    for b = 1:nbins 
        BS_binned(n,b) = nanmean(belief_state(dataSubj.bins == b),1);
    end

%     % Prepare your data
%     choices = dataSubj.choice;%(dataSubj.blocks == 1); % Vector of subject's choices
% 
%     for h = 1:height(dataSubj)
%         if dataSubj.contrast(h) == 0
%             rewards(h) = dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
%         else
%             rewards(h) = 1-dataSubj.recoded_reward(h);%(dataSubj.blocks == 1); % Vector of outcomes
%         end
%     end
%     state = dataSubj.state;%(dataSubj.blocks == 1);
%     condiff = dataSubj.condiff_relative;
%     contrast = dataSubj.contrast;
% 
%     % Set parameter constraints (e.g., learning rate between 0 and 1)
%     params = fit_RL_model(choices, rewards, state, condiff, contrast);
% 
%     alphaParameter(n,1) = params(1);
%     betaParameter(n,1) = params(2);
%     sigmaParameter(n,1) = params(3);
%     ecoperf(n,1) = nanmean(dataSubj.ecoperf);
end

figure

for n = 1:numSubjs

    hold on
    plot(1:nbins,BS_binned(n,:),"Color",[0.5,0.5,0.5],"LineWidth",0.25)
end

hold on
scatter(1:nbins,nanmean(BS_binned),"filled","o",'b',"MarkerFaceAlpha",0.5,"SizeData",60)
hold on
errorbar(1:nbins,nanmean(BS_binned),nanstd(BS_binned)./sqrt(numSubjs),'Color','k')
hold on
plot(1:nbins,nanmean(BS_binned),"Color",'k')
xlabel('Contrast difference bins' ...
    )
ylabel('Belief state - 0.5 for s = 1')

