% Task 1(a)
function yt = ARMA_simulator(T, c, phi, theta, nu, y0)
    % Function to simulate an ARMA(1,1) process with t-distributed innovations
    % The model: yt = c + φ*yt-1 + εt + θ*εt-1, where ε ~ t(ν)
    
    % Preallocate arrays for yt and epsilon (innovations)
    yt = zeros(T, 1);        % Time series output
    epsilon = trnd(nu, T, 1); % t-distributed innovations
    
    % Initialize the process
    yt(1) = y0;               % Initialize with the provided starting value
    epsilon_prev = 0;         % Assume ε_{t-1} is 0 for the first step
    
    % Loop to generate the ARMA process
    for t = 2:T
        % ARMA(1,1) recursion formula
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon_prev;
        
        % Update ε_{t-1} for the next step
        epsilon_prev = epsilon(t);
    end
end

% Task 1 (b)

% Simulate the process with the following parameter specifications

% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Total length of the series
y0 = 40;            % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

% Compute the expected value
expected_value = c / (1 - phi);
disp(['Expected Value: ', num2str(expected_value)]);

% Simulate the ARMA(1,1) process
rng(42);  % Set seed for reproducibility
series = ARMA_simulator(T, c, phi, theta, nu, y0);

% Remove the first 50 observations of our series (burn-in phase)
y = series((burn_in+1):end);
% We should end up with 750 observations

% Task 1(c)
% Create a figure of the realization of the process
figure;
hold on;

% Plot the time series with a specified color and line width
plot(1:(T - burn_in), y, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% Plot the expected value as a horizontal dashed red line
yline(expected_value, '--r', 'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');

% Add title, labels, and grid
title('Simulated ARMA(1,1) Process', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12);
ylabel('Value', 'FontSize', 12);
grid on;

% Customize axes limits
xlim([1 T - burn_in]);
ylim([min(y) - 5, max(y) + 5]);
set(gca, 'FontSize', 12, 'Box', 'on');

% Add legend
legend({'Simulated Process', 'Expected Value'}, 'Location', 'best', 'FontSize', 12);

hold off;