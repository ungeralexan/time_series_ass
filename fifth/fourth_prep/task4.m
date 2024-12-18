%% Task 4

% Add the CML toolbox to the path
addpath('/Users/alexung/Desktop/learning/ATSA/assignments/fifth/CML/');

% Load the ARMA(1,1) realization from Task 2.1
% Replace 'yt' with your actual dataset
y = yt; % Time series data

% Starting values for parameters
x0 = [1.5; 0.75; 0.5; 5]; % [c0, φ0, θ0, ν0]

% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% Call the CML function with Hessian-based covariance matrix
[x, f, g, cov, retcode] = CML(@arma_log_likelihood, @arma_conditional_log_likelihood, y, x0, 1, 1, options);

% Display results
disp('Estimated Parameters:');
disp(['c (constant): ', num2str(x(1))]);
disp(['φ (AR parameter): ', num2str(x(2))]);
disp(['θ (MA parameter): ', num2str(x(3))]);
disp(['ν (degrees of freedom): ', num2str(x(4))]);

disp('Covariance Matrix:');
disp(cov);

disp('Log Likelihood Value at Optimum:');
disp(-f);

%% Task 4.3
% Display parameter estimates
disp('Estimated Parameters and Standard Errors:');

% Extract standard errors from the covariance matrix
std_errors = sqrt(diag(cov));

% Compute 95% confidence intervals
lower_bounds = x - 1.96 * std_errors; % Lower bound
upper_bounds = x + 1.96 * std_errors; % Upper bound

% Display results in a table-like format
param_names = {'c (constant)', 'φ (AR parameter)', 'θ (MA parameter)', 'ν (degrees of freedom)'};
disp('Parameter Estimates, Standard Errors, and 95% Confidence Intervals:');
disp('---------------------------------------------------------------');
disp('Parameter       Estimate       Std. Error       95% CI Lower       95% CI Upper');
disp('---------------------------------------------------------------');
for i = 1:length(x)
    fprintf('%-15s %10.4f %15.4f %15.4f %15.4f\n', ...
        param_names{i}, x(i), std_errors(i), lower_bounds(i), upper_bounds(i));
end
disp('---------------------------------------------------------------');

% Interpretation
disp('Interpretation of Confidence Intervals:');
disp(['If the 95% confidence interval for a parameter does not include a null hypothesis value of interest (e.g., 0 or 1), ' ...
      'we can reject the null hypothesis at the 5% significance level.']);


%% Task 4.4
% Estimated value of φ and its standard error
phi_hat = x(2);                % Extract the estimated φ from results
se_phi = sqrt(cov(2, 2));      % Extract the standard error of φ from the covariance matrix

% Null hypothesis value
phi_null = 0.8;

% Compute the t-statistic
t_stat = (phi_hat - phi_null) / se_phi;

% Display the results
disp('t-Test for H0: φ = 0.8');
disp(['Estimated φ: ', num2str(phi_hat)]);
disp(['Standard Error: ', num2str(se_phi)]);
disp(['t-Statistic: ', num2str(t_stat)]);

% Compare with critical value
critical_value = 1.96; % Two-tailed critical value at 5% significance level
if abs(t_stat) > critical_value
    disp('Result: Reject H0. φ is significantly different from 0.8 at the 5% level.');
else
    disp('Result: Fail to reject H0. φ is not significantly different from 0.8 at the 5% level.');
end


%% Task 4.5
% Compute the p-value
p_value = 2 * (1 - normcdf(abs(t_stat)));

% Display results
disp('Two-Sided p-Value:');
disp(['p-value: ', num2str(p_value)]);

% Interpretation
if p_value < 0.05
    disp('Result: Reject H0. The parameter φ is significantly different from 0.8 at the 5% level.');
else
    disp('Result: Fail to reject H0. The parameter φ is not significantly different from 0.8 at the 5% level.');
end

%% Task 4.6
% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 50000;            % Total length of the series
y0 = 40;            % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

expected_value = c/(1-phi);
disp(expected_value)

% Simulate the ARMA(1,1) process
yt_full = simulate_ARMA_t(T, c, phi, theta, nu, y0);

yt = yt_full((burn_in+1):end);

% Starting values for parameters
x0 = [1.5; 0.75; 0.5; 5]; % [c0, φ0, θ0, ν0]

% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% Call the CML function with Hessian-based covariance matrix
[x, f, g, cov, retcode] = CML(@arma_log_likelihood, @arma_conditional_log_likelihood, yt, x0, 1, 1, options);

% Display results
disp('Estimated Parameters:');
disp(['c (constant): ', num2str(x(1))]);
disp(['φ (AR parameter): ', num2str(x(2))]);
disp(['θ (MA parameter): ', num2str(x(3))]);
disp(['ν (degrees of freedom): ', num2str(x(4))]);

disp('Covariance Matrix:');
disp(cov);

disp('Log Likelihood Value at Optimum:');
disp(-f);