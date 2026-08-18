clc
clearvars
x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
betas = [0.3,0.6,1,2,3];
colors = winter(5);

figure
hold on
for b = 1:length(betas)
    % Convert beta to equivalent strength: strength = (1-beta)/beta
    equivalent_strength = (1 - betas(b)) / betas(b);
    
    yPG = anti_sigmoidPG(x, betas(b));
    % yRB = odds_transformRB(x, betas(b));
    
    plot(x, yPG, 'Color', colors(b,:), 'LineStyle', '-', 'LineWidth', 2);
    % plot(x, yRB, 'Color', colors(b,:), 'LineStyle', '--', 'LineWidth', 2);
end
plot(x, x, 'k', 'LineWidth', 1);