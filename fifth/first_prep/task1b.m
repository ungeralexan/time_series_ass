% Parameters
T = 800;           % Total sample size
burn_in = 50;      % Number of observations to discard
c = 2;             % Constant term
phi = 0.95;        % AR parameter
theta = 0.25;      % MA parameter
nu = 4;            % Degrees of freedom for t-distribution

% Compute the expected value of the process
expected_value = c / (1 - phi);

% Simulate the process
yt_full = simulate_ARMA11(T, c, phi, theta, nu, expected_value);

% Discard the first 50 observations
yt = yt_full(burn_in+1:end);


% Calculate the mean (stationary assumption)
mu = 0; % Replace with calculated mean if there's an intercept term

% Plotting
figure;
plot(yt, 'LineWidth', 1.2);
hold on;
yline(mu, '--r', 'LineWidth', 1.5, 'DisplayName', 'Mean'); % Add mean line
title('ARMA(1,1) Process', 'FontSize', 14);
xlabel('Time', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
legend('ARMA Process', 'Mean', 'FontSize', 12);
grid on;
set(gca, 'FontSize', 12);