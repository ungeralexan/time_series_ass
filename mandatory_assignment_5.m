%% 1.1. Define a function to simulate ARMA(1,1) with t-distributed innovations

function y = simulate_ARMA11_t(T, c, phi, theta, nu, y0)
    % This function simulates an ARMA(1,1) process:
    %   y_t = c + phi * y_(t-1) + eps_t + theta * eps_(t-1)
    % with eps_t ~ t(nu).
    %
    % Inputs:
    %   T    - Sample size
    %   c    - Constant term
    %   phi  - AR(1) coefficient
    %   theta- MA(1) coefficient
    %   nu   - Degrees of freedom for t-distribution
    %   y0   - Initial value for y_(0)
    %
    % Output:
    %   y    - Simulated time series (T x 1)

    % Preallocate
    y = zeros(T,1);
    epsilons = trnd(nu, T, 1); % Draw from t-distribution
    y(1) = y0;
    eps_lag = 0; % Start with eps_(0) = 0 for convenience

    for t = 2:T
        y(t) = c + phi * y(t-1) + epsilons(t) + theta * eps_lag;
        eps_lag = epsilons(t);
    end
end

%% 1.2. Call the function and discard burn-in

% Parameters
c = 2;
phi = 0.95;
theta = 0.25;
nu = 4;
T = 800;

% Compute the expected value for initialization:
y0 = c/(1 - phi);

% Simulate the process
rng(500)
y_full = simulate_ARMA11_t(T, c, phi, theta, nu, y0);

% Discard first 50 observations
y = y_full(51:end);

figure('Position',[100 100 800 400]); 
hSeries = plot(y, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]); 
grid on; hold on;

% Add a line representing the mean of the process and store its handle
hMeanLine = yline(y0, '--r', 'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom', ...
    'FontSize', 10, 'FontName', 'Arial');

xlabel('Time', 'FontSize', 12, 'FontName', 'Arial', 'FontWeight', 'bold');
ylabel('Realisation', 'FontSize', 12, 'FontName', 'Arial', 'FontWeight', 'bold');
title('Simulated ARMA(1,1) Process', ...
    'FontSize', 14, 'FontName', 'Arial', 'FontWeight', 'bold');

% Adjust axes properties
ax = gca;
ax.FontSize = 10;
ax.FontName = 'Arial';
ax.LineWidth = 1;
ax.Box = 'on';
ax.XMinorGrid = 'on';
ax.YMinorGrid = 'on';

% Add a legend
legend([hSeries hMeanLine], {'Simulated Series', 'Mean'}, 'FontSize',10, 'FontName','Arial', 'Location','best');