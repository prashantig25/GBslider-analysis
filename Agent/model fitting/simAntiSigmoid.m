clc
clearvars

betas = [0.2,0.5,1,1.5,2];
x = linspace(0,1,11);
colors = copper(6);
figure
hold on
plot(x,x,'Color','k','LineWidth',2)
for b = 1:length(betas)
    y = anti_sigmoidPG(x,betas(b));

    hold on
    plot(x,y,'Color',colors(b,:),'LineWidth',2)
end
xlabel('Actual value')
ylabel('Transformed value')
legend('Original','Beta = 0.2','0.5','1','1.5','2','Location','best')