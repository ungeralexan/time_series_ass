%% ATS Mandatory Assignment 05

% **Assignment 5 (Mandatory)**  
% **MatNr:6940188**
% **MatNr:6946672**
% **Submission Date:** December 12th, 2025

%%
%% We begin by activating the folder path
% We contructed this statement to check the folder path and to update it to be
% able to run the script
if isdeployed
    scriptFolder = ctfroot; 
else
    scriptFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end

if isempty(scriptFolder)
    error('Could not determine the folder of the currently running script.');
end

cd(scriptFolder);

disp(['Current folder changed to: ', scriptFolder]);
%% Task 1: Simulating an ARMA(1,1) Process

%%% Task 1.1: We define the ARMA Simulator Function

function yt = ARMA_simulator(T, c, phi, theta, nu, y0)
    yt = zeros(T, 1); 
    epsilon = trnd(nu, T, 1); 
    yt(1) = y0;
    epsilon_prev = 0;
    for t = 2:T
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon_prev;
        epsilon_prev = epsilon(t); 
    end
end

%% Task 1.2: We will now simulate the process
% Parameters
c = 2;              
phi = 0.95;         
theta = 0.25;      
nu = 4;             
T = 800;            
y_init = 40;            
burn = 50;      

% First we compute the expected value of the process
expected_value = c / (1 - phi);
disp(['Expected Value: ', num2str(expected_value)]);

% Here we use the above defined ARMA_simulator function to generate the time series
rng(42);  % setting seed for reproducibility to ensure consistent results
series = ARMA_simulator(T, c, phi, theta, nu, y_init);

% We remove burn-in phase (of 50 observations)
y = series((burn + 1):end);

% We here store the series in order to come back at it in task 5 when we
% have to simulate the process again
ysafe = y;

%% Task 1.3: Here we visualize the simulated ARMA(1,1) process
figure; 
hold on; 
plot(1:(T - burn), y, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);
yline(expected_value, '--r', 'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');
title('Simulated ARMA(1,1) Process', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12); 
ylabel('Value', 'FontSize', 12); 
grid on;
xlim([1 T - burn]); 
ylim([min(y) - 5, max(y) + 5]);
legend({'Simulated Process', 'Expected Value'}, 'Location', 'best', 'FontSize', 12);
set(gca, 'FontSize', 12, 'Box', 'on'); 
hold off;
%% Task 2: Unit Root Test

% Task 2.2: We estimate the series with OLS therefore:

% We define our dependent Variable
yt = y(2:end);

% Now we define our lagged Variable
yt_lag = y(1:end-1);

% We construct our design matrix
X = [ones(length(yt_lag), 1), yt_lag];

% We now compute our OLS estimates
beta = (X' * X) \ (X' * yt);

% We extract our two parameters
alpha_hat = beta(1); 
rho_hat = beta(2);

% Our predictions and residuals
y_pred = X * beta;          
residuals = yt - y_pred;

% We estimate our sigma^2
sigma_squared = (1 / (length(yt) - 2)) * sum(residuals .^ 2);

% Covariance matrix
Cov_beta = sigma_squared * inv(X' * X);

% Standard errors
s_errors = sqrt(diag(Cov_beta));
s_error_alpha = s_errors(1); 
s_error_rho = s_errors(2);  

disp('---------The results for Task 2.2:');
disp('Estimated coefficients (βˆ):');
disp(['c (constant): ', num2str(beta(1))]); 
disp(['φ (AR parameter): ', num2str(beta(2))]); 
disp('---------The results for Task 2.3:');
disp('Standard Errors (s.e.):');
disp(['s.e.(c): ', num2str(s_error_alpha)]); 
disp(['s.e.(φ): ', num2str(s_error_rho)]);
%% Task 2.4: Next we perform Dickey-Fuller test (unit root or not)

% We calculate t-statistic
phi_hat = beta(2);         
se_phi = s_errors(2);   
t_stat = (phi_hat - 1) / se_phi;
disp('---------The results for Task 2.4:');
disp(['t-statistic: ', num2str(t_stat)]);
% Comparing with critical value
critical_value = -2.86; 

if t_stat < critical_value
    disp('We reject the null hypothesis (H0: φ = 1). The series is stationary.');
else
    disp('We fail to reject the null hypothesis (H0: φ = 1). The series is non-stationary.');
end
%% Task 3: Constructing the Conditional Log Likelihood

%%% Task 3.1
% We first define a function that constructs the conditional maximum
% likelihood contributions

% First the function that calculates the individual contributions
function logL_contributions = ml_contributions(params, y)
    % We first extract parameters
    c = params(1);
    phi = params(2);
    theta = params(3);
    nu = params(4);
    
    % Then we initialize variables
    T = length(y);
    epsilons = zeros(T, 1); 
    logL_contributions = zeros(T-1, 1);
    
    % Next we compute the series of innovations ε_t
    for t = 2:T
        epsilons(t) = y(t) - c - phi * y(t-1) - theta * epsilons(t-1);
    end

    % Dropping ε1 (the first innovation)
    epsilons = epsilons(2:end);
    
    % Computing log likelihood contributions
    for t = 1:(T-1)
        logL_contributions(t) = gammaln((nu + 1) / 2) ...
                                - gammaln(nu / 2) ...
                                - 0.5 * log(nu * pi) ...
                                - (nu + 1) / 2 * log(1 + (epsilons(t)^2 / nu));
    end
end


% Now we define our second function that sums all the individual
% contributions 
% 2nd helper function: loglikelihood 
function total_logL = loglikelihood(params, y)
    logL_contributions = ml_contributions(params, y);

    % Total log likelihood: summing the contributions
    total_logL = sum(logL_contributions);
end
%% Task 3.2
% Case (a) parameters (these are the true parameters)
params_a = [2; 0.95; 0.25; 4];

% We define a display for our results
disp('---------The results for Task 3.2:');


% Now we compute the total log likelihood for our case A
case_a = loglikelihood(params_a, y);

disp(['Total conditional log likelihood for case a): ', num2str(case_a)]);
% Case (b) parameters
params_b = [1.5; 0.75; 0.5; 6];

% We compute the total log likelihood for case (b)
case_b = loglikelihood(params_b, y);

disp(['Total conditional log likelihood for case b): ', num2str(case_b)]);
%% Task 4: Parameter Estimation with Maximum Likelihood

%% Task 4.1
% We first define the neg_loglikelihood function, which computes the negative 
% total conditional log likelihood for our ARMA(1,1) series
function negLogL = neg_loglikelihood(params, y)
    logL_contributions = ml_contributions(params, y);
    negLogL = -sum(logL_contributions);
end


%% CML toolbox
% This is a very important section as we activate the CML folder, we
% constructed this script on a windows operating system so to be able to
% run it on a different operating system, 
% you might need to change '\' to '/'!
addpath(['CML', filesep, 'CML']);

%% Task 4.2

% We define the starting values
starters = [1.5; 0.75; 0.5; 5];

% Defining optimization settings for fminsearch algorithm
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% We now estimate parameters by minimizing the negative log likelihood
[x, f, g, cov, retcode] =             CML(@neg_loglikelihood,   ...  
                              @ml_contributions,    ...  
                              y,                    ...  
                              starters,             ...  
                              1,                    ...  
                              1,                    ...  
                              options);             ...  

% Display our results
disp('---------The results for Task 4.2:');
disp('Optimized parameters = ');
disp(x);
disp('Minimum negative log-likelihood = ');
disp(f);
disp('Value of the gradient at x=');
disp(g);
disp('Covariance matrix of estimates = ');
disp(cov);
disp('Exit condition of the optimization function used=');
disp(retcode)
% We store the optimized parameters and covariance matrix (as we need them
% for the following tasks
paramEst = x;      
covEst = cov;      

%% Task 4.3
% Here we display the standard errors for our estimates
SE = sqrt(diag(covEst));

% Constructing 95% confidence intervals
crit_value = 1.96; 
CI_lower = paramEst - crit_value .* SE;
CI_upper = paramEst + crit_value .* SE;


disp('---------The results for Task 4.3:');
paramNames = {'c', 'phi', 'theta', 'nu'};
fprintf('\nParameter Estimates with 95%% CIs:\n');
for i = 1:length(paramEst)
    fprintf('%s:  SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, SE(i), CI_lower(i), CI_upper(i));
end
%% Task 4.4: Perform t-test for H0: φ = 0.8
% We test the null hypothesis H0: φ = 0.8 at a 5% significance level (two-sided test)
phiEst = paramEst(2); 
phi_standard_error = SE(2);        

% Now we compute the t-statistic
tStat = (phiEst - 0.8) / phi_standard_error;

% Again our critical value we defined before
crit_value = 1.96;

disp('---------The results for Task 4.4:');
fprintf('\nT-test for H0: phi = 0.8\n');
fprintf('   Estimate of phi = %.4f\n', phiEst);
fprintf('   Standard Error   = %.4f\n', phi_standard_error);
fprintf('   Test statistic   = %.4f\n', tStat);
if abs(tStat) > crit_value
    fprintf('   => Reject H0 at the 5%% level (|tStat| > 1.96).\n\n');
else
    fprintf('   => Fail to reject H0 at the 5%% level.\n\n');
end

%% Task 4.5: computing two-Sided p-value
pValue = 2 * (1 - normcdf(abs(tStat))); 
disp('---------The results for Task 4.5:');
fprintf('Test statistic for H0: phi = 0.8 is tStat = %.4f\n', tStat);
fprintf('Two-sided p-value = %.6f\n', pValue);
if pValue < 0.05
    fprintf('=> p-value < 0.05 => Reject H0 at 5%% level.\n\n');
else
    fprintf('=> p-value >= 0.05 => Fail to reject H0 at 5%% level.\n\n');
end

%% Task 4.6: here we simulate a longer ARMA(1,1) series (T = 50,000) and re-estimate parameters

% First again we define our parameters (we just change the T)
           
T = 50000;          


% Simulating the ARMA(1,1) process
series = ARMA_simulator(T, c, phi, theta, nu, y_init);

% Updating the series
y = series((burn+1):end);

% Now we will re-estimate parameters
[x, f, g, cov, retcode] = CML(@neg_loglikelihood,   ...
                              @ml_contributions,    ...
                              y,                    ...
                              starters,             ...
                              1,                    ...
                              1,                    ...
                              options);

disp('---------The results for Task 4.6:');
disp('Optimized parameters = ');
disp(x);
disp('Minimum negative log-likelihood = ');
disp(f);
disp('Gradient at optimum = ');
disp(g);
disp('Covariance matrix of estimates = ');
disp(cov);

disp(['Exit flag from fminsearch = ', num2str(retcode)]);

% We construct confidence intervals for new estimates
paramEst = x;      
covEst = cov;      
SE = sqrt(diag(covEst)); 
CI_lower = paramEst - crit_value .* SE;
CI_upper = paramEst + crit_value .* SE;

fprintf('\nParameter Estimates with 95%% CIs (T = 49,750):\n');
for i = 1:length(paramEst)
    fprintf('%s:  SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, SE(i), CI_lower(i), CI_upper(i));
end

% Lets perform our two sided t-test for H0: phi = 0.8 at 5% significance

phiEst = paramEst(2);
phiSE  = SE(2);

% Test statistic:
tStat = (phiEst - 0.8) / phiSE;

% Critical value at alpha=5% (two-sided) from standard normal is about +-1.96
critVal = 1.96;

fprintf('\nT-test for H0: phi = 0.8\n');
fprintf('   Estimate of phi = %.4f\n', phiEst);
fprintf('   Standard Error   = %.4f\n', phiSE);
fprintf('   Test statistic   = %.4f\n', tStat);

if abs(tStat) > critVal
    fprintf('   => We reject H0 at the 5%% level (|tStat| > 1.96).\n\n');
else
    fprintf('   => We fail to reject H0 at the 5%% level.\n\n');
end

% Now we construct the p-value 
pValue = 2 * (1 - normcdf(abs(tStat))); 

fprintf('Test statistic for H0: phi = 0.8 is tStat = %.4f\n', tStat);
fprintf('Two-sided p-value = %.6f\n', pValue);

% Interpretation:
if pValue < 0.05
    fprintf('=> p-value < 0.05 => We reject H0 at 5%% level.\n\n');
else
    fprintf('=> p-value >= 0.05 => We fail to reject H0 at 5%% level.\n\n');
end

%% Task 5: Quasi-Maximum Likelihood
% 5.1

% 1st helper function for task 5: qml_contributions (here we calculate the quasi-maximum likelihood contributions)
function logL_contributions = qml_contributions(params, y)
    c      = params(1);
    phi    = params(2);
    theta  = params(3);
    sigma2 = params(4);

    T = length(y);
    epsilons = zeros(T, 1);

    % We compute the series of innovations
    for t = 2:T
        epsilons(t) = y(t) - c - phi*y(t-1) - theta*epsilons(t-1);
    end

    % Dropping the first innovation
    epsilons = epsilons(2:end); 

    % Next we compute log-likelihood contributions for Gaussian
    logL_contributions = -0.5 * log(2*pi*sigma2) ...
                         - 0.5 * (epsilons.^2 ./ sigma2);
end



% Now we again define a function that sums all these individual
% contributions together
function total_logL = qml_l(params, y)
   
    % Contributions from the helper function
    logL_contributions = qml_contributions(params, y);

    % Summing over t = 2..T
    total_logL = -sum(logL_contributions);
end

%% 5.3
% Now we come back to the series we stored in the first task which had the
% length of 750
y = ysafe;

% Starting values
starters_qml = [1.5; 0.75; 0.5; 1]; 

% Optimization settings
options = optimset('Display', 'iter', ...
                   'TolX', 1e-40, ...
                   'TolFun', 1e-40, ...
                   'MaxIter', 1e10, ...
                   'MaxFunEvals', 100000);

   

% Running the CML Toolbox
[x_qml, fval_qml, g_qml, cov_qml, retcode_qml] = ...
    CML(@qml_l,                    ...  
        @qml_contributions,        ...  
        y,                         ...  
        starters_qml,              ...  
        1,                         ... 
        3,                         ...  
        options);                  ...  

% Results of the QML estimation
disp('---------The results for Task 5.3:');
fprintf('QML results (Gaussian assumption) for ARMA(1,1):\n');
fprintf('Estimated parameters:\n');
disp(x_qml);
fprintf('Negative Log-Likelihood at optimum = %.4f\n', fval_qml);
fprintf('Gradient at optimum:\n');
disp(g_qml);
fprintf('Covariance matrix (QML-based):\n');
disp(cov_qml);
fprintf('Exit flag: %d\n', retcode_qml);

%% Task 5.4
paramEstQML = x_qml;      
covEstQML   = cov_qml;   

% Standard errors
SE_qml = sqrt(diag(covEstQML));

% 95% confidence intervals
CI_lower_qml = paramEstQML - crit_value .* SE_qml;
CI_upper_qml = paramEstQML + crit_value .* SE_qml;

paramNames = {'c','phi','theta','sigma^2'};
fprintf('\nParameter Estimates with 95%% CIs:\n');
for i = 1:length(paramEstQML)
    fprintf('%s: SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, SE_qml(i), CI_lower_qml(i), CI_upper_qml(i));
end
%% Task 6: Comparing ML and QML through various realizations of the process

% Task 6.1.
% Parameters from task we define them again for the loop
   
% We only change the sample size to as the rest parameters stay the same
T = 800;    


% We define the number of realizations of the process
K= 2500;       

% For storing estimates
ML_estimates = zeros(K, 4);       
QML_estimates = zeros(K, 4);      
ML_covariances = zeros(K, 4, 4);  
ML_standard_errors = zeros(K, 4); 

% we define the starting parameters for both again 
starters_ML = [1.5; 0.75; 0.5; 5];  
starters_QML = [1.5; 0.75; 0.5; 1]; 

% We set the optimization settings
options = optimset('Display', 'off', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);
  

% we set a seed for reproducibility of consistent results
rng(42)

% Looping Over K = 2500 simulations 
for k = 1:K
    % First we simulate ARMA(1,1) process
    series = ARMA_simulator(T, c, phi, theta, nu, y_init);
    y = series((burn + 1):end);  

    % Then we estimate parameters with ML
    [ML_params, ~, ~, ML_cov, ~] = CML(@neg_loglikelihood, ...
                                       @ml_contributions, ...
                                       y, starters_ML, 1, 1, options);
    ML_estimates(k, :) = ML_params';               
    ML_covariances(k, :, :) = ML_cov;              
    ML_standard_errors(k, :) = sqrt(diag(ML_cov))';

    % Finally we estimate parameters with QML (Gaussian)
    [QML_params, ~, ~, ~, ~] = CML(@qml_l, ...
                                   @qml_contributions, ...
                                   y, starters_QML, 1, 3, options);
    QML_estimates(k, :) = QML_params';             
end

disp('---------The results for Task 6.1:');
disp('Simulation and estimation complete.');
disp('ML estimates, QML estimates, and ML standard errors stored.');
%% Task 6.2: we compute kernel densities and visualize the densities of ML and QML estimators

% These are the parameters we want to plot for
params = {'c', '\phi', '\theta'};  

% We define teh colurs for our plots
colors = {'b', 'r'};      

% We will use straight lines
lineStyles = {'-'};   

% Plotting
figure('Color', 'w', 'Position', [100, 100, 1400, 600]); 

t = tiledlayout(1, 3, 'TileSpacing', 'Compact', 'Padding', 'Compact');

for i = 1:3  
    nexttile;  

    % Now we compute kernel densities
    [ML_density, ML_x] = ksdensity(ML_estimates(:, i));   
    [QML_density, QML_x] = ksdensity(QML_estimates(:, i));

    % Now we plot densities of corresponding ML and QML estimators
    plot(ML_x, ML_density, 'LineWidth', 2, 'Color', colors{1}, 'LineStyle', lineStyles{1});
    hold on;

    plot(QML_x, QML_density, 'LineWidth', 2, 'Color', colors{2}, 'LineStyle', lineStyles{1});

    title(['Kernel Density of ', params{i}], 'FontSize', 14, 'FontWeight', 'bold'); 
    xlabel(params{i}, 'FontSize', 12, 'FontWeight', 'bold'); 
    ylabel('Density', 'FontSize', 12, 'FontWeight', 'bold'); 

    legend({'ML', 'QML'}, 'Location', 'Best', 'FontSize', 12, 'Color', 'none');

    grid on;
    grid minor;

    hold off;
end

sg = sgtitle('Kernel Densities of ML and QML Estimates', 'FontSize', 16, 'FontWeight', 'bold');
current_pos = sg.Position;
sg.Position = [current_pos(1), current_pos(2) + 0.05, current_pos(3)];

%% Task 7: Confidence Interval Analysis

% Now we will compute 95% confidence intervals for ML estimates across K simulations 
% and determine the proportion of true parameters falling within these intervals


% --- True parameter values
true_params = [2; 0.95; 0.25; 4];  % [c, phi, theta, nu]

% --- Number of ensembles
K = size(ML_estimates, 1);  % Should already be 2500 from Task 6.1

% --- Compute 95% Confidence Intervals for ML estimates
crit_value = 1.96;  % For 95% CI
ML_CI_lower = ML_estimates - crit_value .* ML_standard_errors;  % Lower bounds
ML_CI_upper = ML_estimates + crit_value .* ML_standard_errors;  % Upper bounds

% --- Test whether true values lie within the bounds
coverage = zeros(1, 4);  % To store the coverage probabilities for [c, phi, theta, nu]

for i = 1:4  % Loop over parameters (c=1, phi=2, theta=3, nu=4)
    % Check if true parameter value lies within the CI for each ensemble
    coverage(i) = mean((true_params(i) >= ML_CI_lower(:, i)) & ...
                       (true_params(i) <= ML_CI_upper(:, i)));
end

% --- Improved printing of coverage results
param_labels = {'c', 'phi', 'theta', 'nu'};  % Parameter names as cell array of strings
fprintf('Coverage Results (Proportion of 95%% CIs containing true values):\n');
fprintf('--------------------------------------------------------------\n');
for i = 1:4
    fprintf('  Parameter %-6s: %.2f%%\n', param_labels{i}, coverage(i) * 100);
end
fprintf('--------------------------------------------------------------\n');

