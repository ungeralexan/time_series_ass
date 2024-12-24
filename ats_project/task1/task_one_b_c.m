%% This file will yield task b & c

%% Task b
% simulate the process for the following parameter specification

% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Total length of the series
y0 = 40;            % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

% we compute the expected value
expected_value = c/(1-phi);
disp(expected_value);

% Simulate the ARMA(1,1) process
rng(200)

% now we simulate our series (one realization of the proces
% Fix a seed so we are able to reproduce the realization of the process
rng(42);
series = ARMA_simulator(T, c, phi, theta, nu, y0);

% Remove the first 50 observations of our series (burn in phase9
y = series((burn_in+1):end);
% we should end up with 750 observations


%% Task c
% Now I will be creating a figure of the realiztaion of the process
figure;
hold on;

% Plot the time series (with a color)
plot(1:(T-burn_in), y, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% Plot the expected value as a horizontal line (therefore we calculated it)
yline(expected_value, '--r', 'LineWidth', 1.5, 'Label', 'Expected Value', ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');

% Add title, labels, and grid
title('Simulated ARMA(1,1) Process', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
grid on;

% Customize axes
xlim([1 T-burn_in]);
ylim([min(y)-5, max(y)+5]);
set(gca, 'FontSize', 12, 'Box', 'on');

% Add legend
legend({'Simulated Process', 'Expected Value'}, 'Location', 'best', 'FontSize', 12);

hold off;