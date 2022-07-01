q = [0.1 0.3 0.5 0.8 1.0];
maxFast = [0.001128 0.002253 0.293219 0.998848 0.936943];
maxSimple = [0.010514 0.015676 0.722930 0.999759 0.943034];

plot(q, maxFast)
title('Maximum error')
xlabel('q')
ylabel('Group distance')

hold on;

plot(q, maxSimple)
title('Maximum error')
xlabel('q')
ylabel('Group distance')

legend('Original', 'Simple')