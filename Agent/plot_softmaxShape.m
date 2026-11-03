% MATLAB example
beta = importdata("betaNoReward_simpleRL.mat");%unifrnd(1,20,[10,1]);  % Inverse temperatures
x = linspace(0.5, 1, 100);

% Create blue colormap
n_lines = length(beta);
blue_colors = winter(n_lines);

figure;
hold on;
for b = 1:length(beta)
    probs = exp(b * x) ./ sum(exp(b * x), 2);
    plot(x, probs, 'LineWidth', 1.5, 'DisplayName', ['β = ' num2str(beta(b))],'Color',blue_colors(b,:));
end
xlabel('Q-value Difference (Q₁ - Q₂)');
ylabel('Probability');
title('Softmax Action Selection');
% legend show;
grid on;
