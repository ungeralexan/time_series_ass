%% Task 5
% Add the CML toolbox to the path
addpath('CML');

% Load the ARMA(1,1) realization (replace 'yt' with your actual dataset)
y = yt; % Time series data from Task 1.1

% Starting values for parameters: [c0, φ0, θ0, σ0]
x0 = [1.5; 0.75; 0.5; 1]; % Initial guesses: [c, φ, θ, σ]

% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% Call the CML function using QML covariance matrix (covPar = 3)
[x, f, g, cov, retcode] = CML(@arma_qml_log_likelihood, @arma_qml_log_likelihood_contributions, ...
                              y, x0, 1, 3, options);

% Display results
disp('QML Estimation Results for ARMA(1,1):');
disp('-----------------------------------------');
disp(['c (constant): ', num2str(x(1))]);
disp(['φ (AR parameter): ', num2str(x(2))]);
disp(['θ (MA parameter): ', num2str(x(3))]);
disp(['σ (standard deviation): ', num2str(x(4))]);

disp('QML Covariance Matrix:');
disp(cov);

disp('Log Likelihood Value at Optimum:');
disp(-f);
disp('-----------------------------------------');


%% Task 5.4
% Display QML Estimation Results
disp('QML Estimation Results for ARMA(1,1):');
disp('---------------------------------------------------------------');

% Extract parameter estimates and standard errors
estimates = x;                     % Parameter estimates: c, φ, θ, σ
std_errors = sqrt(diag(cov));      % Standard errors from covariance matrix

% Compute 95% confidence intervals
lower_bounds = estimates - 1.96 * std_errors; % Lower bound
upper_bounds = estimates + 1.96 * std_errors; % Upper bound

% Parameter names
param_names = {'c (constant)', 'φ (AR parameter)', 'θ (MA parameter)', 'σ (std deviation)'};

% Print results in table format
disp('Parameter       Estimate       Std. Error       95% CI Lower       95% CI Upper');
disp('---------------------------------------------------------------');
for i = 1:length(estimates)
    fprintf('%-15s %10.4f %15.4f %15.4f %15.4f\n', ...
        param_names{i}, estimates(i), std_errors(i), lower_bounds(i), upper_bounds(i));
end
disp('---------------------------------------------------------------');

% Log likelihood value
disp(['Log Likelihood Value at Optimum: ', num2str(-f)]);

% Comparison Interpretation
disp('Comparison to ML Estimation (Task 4):');
disp(['1. Parameter estimates (c, φ, θ, σ) are now based on QML assumptions, ' ...
      'which assumes Gaussian innovations.']);
disp(['2. Standard errors may differ due to the QML covariance matrix, ' ...
      'which accounts for model misspecification.']);
disp(['3. Differences in confidence intervals reflect the robustness of QML, ' ...
      'especially if the true innovation distribution deviates from Gaussian assumptions.']);
