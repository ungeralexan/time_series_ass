%% Task 4 revisted
% Add the CML toolbox to the path
addpath('/Users/alexung/Desktop/learning/ATSA/assignments/fifth/repast');
addpath('/Users/alexung/Desktop/learning/ATSA/assignments/fifth/first_prep');
addpath('/Users/alexung/Desktop/learning/ATSA/assignments/fifth/CML/');

% our series is y with 750 observations

% Starting values for parameters
x0 = [1.5; 0.75; 0.5; 5]; % [c0, φ0, θ0, ν0]

% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% important how do my functions are called
% 1) arma_conditional_log_likelihood is the function calculates the conditional
%likelihoods

% 2) arma_log_likelihood is the function that calculates the negative
% likelihood


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

%my results I dont know yet if I am happy with them 

%{
Estimated Parameters:
c (constant): 2.6794
φ (AR parameter): 0.93418
θ (MA parameter): 0.23437
ν (degrees of freedom): 4.2977

Log Likelihood Value at Optimum:
  -1.2576e+03
%}

%% Compute standard errors for my estimates
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

% For all of those values in the CI we cannot reject the Nullhypothesis
% that it is the true value of the process

%% Task 4.4
% Estimated value of φ and its standard error
% We are still in stationary world do not forget this
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
% two sided t-test
% Compute the p-value
p_value = 2 * (1 - normcdf(abs(t_stat)));

% Display results
disp('Two-Sided p-Value:');
disp(['p-value: ', num2str(p_value)]);

% p value of null means that we have to reject the nullhypothesis
% Interpretation
if p_value < 0.05
    disp('Result: Reject H0. The parameter φ is significantly different from 0.8 at the 5% level.');
else
    disp('Result: Fail to reject H0. The parameter φ is not significantly different from 0.8 at the 5% level.');
end


%%
% Now lets do the same but with 50000 simulations


%%
%%Simualte the process 50000 times
% we still include the burn in stuff so we loose 50 observations
% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 50000;            % Total length of the series
y0 = c/(1-phi);     % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

disp(expected_value)


% Simulate the ARMA(1,1) process that creates a series of 800 values
rng(200)
series= simulate_ARMA_t(T, c, phi, theta, nu, y0);



% Remove the first 50 observations
y = series((burn_in+1):end); % works as we have 750 length vector
%y now has 49950 observations

%%
% Now lets first lets find the best estimated for my process
% our series is y with 750 observations

% Starting values for parameters
x0 = [1.5; 0.75; 0.5; 5]; % [c0, φ0, θ0, ν0]

% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% important how do my functions are called
% 1) arma_conditional_log_likelihood is the function calculates the conditional
%likelihoods

% 2) arma_log_likelihood is the function that calculates the negative
% likelihood


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

% I would say my parameter clearly improved towards the real parameters:
%{
Estimated Parameters:
c (constant): 1.9638
φ (AR parameter): 0.95099
θ (MA parameter): 0.24265
ν (degrees of freedom): 3.9736 
Log Likelihood Value at Optimum:
  -8.4208e+04
%}

%% Now test our Nullhypothesis for 0.8
% Estimated value of φ and its standard error
% We are still in stationary world do not forget this
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

% We obviously do not fail to reject the nullhypothesis

%% What about for a two sided test 
% two sided t-test
% Compute the p-value
p_value = 2 * (1 - normcdf(abs(t_stat)));

% Display results
disp('Two-Sided p-Value:');
disp(['p-value: ', num2str(p_value)]);

% p value of null means that we have to reject the nullhypothesis
% Interpretation
if p_value < 0.05
    disp('Result: Reject H0. The parameter φ is significantly different from 0.8 at the 5% level.');
else
    disp('Result: Fail to reject H0. The parameter φ is not significantly different from 0.8 at the 5% level.');
end