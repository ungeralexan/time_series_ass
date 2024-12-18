% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Total length of the series
y0 = 40;            % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

expected_value = c/(1-phi);
disp(expected_value)

% Simulate the ARMA(1,1) process
rng(200)
yt_full = simulate_ARMA_t(T, c, phi, theta, nu, y0);

% Remove the first 50 observations
yt = yt_full((burn_in+1):end);


% Create a fancy plot
figure;
hold on;

% Plot the time series
plot(1:(T-burn_in), yt, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% Plot the expected value as a horizontal line
yline(expected_value, '--r', 'LineWidth', 1.5, 'Label', 'Expected Value', ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');

% Add title, labels, and grid
title('Simulated ARMA(1,1) Process with Expected Value', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
grid on;

% Customize axes
xlim([1 T-burn_in]);
ylim([min(yt)-5, max(yt)+5]);
set(gca, 'FontSize', 12, 'Box', 'on');

% Add legend
legend({'Simulated Process', 'Expected Value'}, 'Location', 'best', 'FontSize', 12);

hold off;