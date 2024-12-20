%%Task 5 redonde 

% first again make sure you have 750 observations
% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Total length of the series
y0 = c/(1-phi);     % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

disp(expected_value)


% Simulate the ARMA(1,1) process that creates a series of 800 values
rng(200)
series= simulate_ARMA_t(T, c, phi, theta, nu, y0);



% Remove the first 50 observations
y = series((burn_in+1):end); % works as we have 750 length vector


% Now using Quasi Maximum Likelihood I am going to try to estimate the best
% parameter set

% Starting values for parameters: [c0, φ0, θ0, σ0]
x0 = [1.5; 0.75; 0.5; 1]; % Initial guesses: [c, φ, θ, σ^2]

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
disp(['σ (Variance): ', num2str(x(4))]);

disp('QML Covariance Matrix:');
disp(cov);

disp('Log Likelihood Value at Optimum:');
disp(-f);
disp('-----------------------------------------');

%{
My resulting parameters are
Log Likelihood Value at Optimum:
  -1.2939e+03
QML Estimation Results for ARMA(1,1):
-----------------------------------------
c (constant): 2.8716
φ (AR parameter): 0.9291
θ (MA parameter): 0.24375
σ (Variance): 1.3614
%}


%% Task 4 estimates standard errors and confidence intervalls

disp('QML Estimation Results for ARMA(1,1):');
disp('---------------------------------------------------------------');

% Extract parameter estimates and standard errors
estimates = x;                     % Parameter estimates: c, φ, θ, σ^2
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
