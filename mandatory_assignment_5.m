%% 
% 1. Simulating an ARMA(1,1)

% Given parameters
c = 2;
phi = 0.95;
theta = 0.25;
nu = 4;
T = 800;

% Compute the starting value from the unconditional mean
y0 = c/(1 - phi);

% Simulate the ARMA(1,1) with t-distributed innovations
y = simulate_arma_t(T, c, phi, theta, nu, y0);

% Discard the first 50 observations (burn-in period)
burn_in = 50;
y_final = y((burn_in+1):end);

% Create a professional plot
figure;
plot(y_final, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]); % nice blue line
grid on;
title('Simulated ARMA(1,1) Process with t-Distributed Innovations', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12);
ylabel('y_t', 'FontSize', 12);
set(gca, 'FontSize', 12, 'Box', 'on', 'LineWidth', 1);


% 1.1.
