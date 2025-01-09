%% ATS Mandatory Assignment

%% First I activate the folder path
% Get the full path of the currently running script
if isdeployed
    % For deployed applications
    scriptFolder = ctfroot; % Base folder for compiled code
else
    % For MATLAB scripts
    scriptFolder = fileparts(matlab.desktop.editor.getActiveFilename);
end

% Check if scriptFolder was successfully retrieved
if isempty(scriptFolder)
    error('Could not determine the folder of the currently running script.');
end

% Change the current folder to the script's folder
cd(scriptFolder);

% Optional: Display the current folder to confirm the change
disp(['Current folder changed to: ', scriptFolder]);

%% Task 1a) we create the function that simulates tha ARMA(1,)


function yt = ARMA_simulator(T, c, phi, theta, nu, y0)


    % --- Initialize ---
    yt = zeros(T, 1); % Preallocate memory for the time series
    epsilon = trnd(nu, T, 1); % Generate t-distributed innovations
    yt(1) = y0; % Set initial value
    epsilon_prev = 0; % Initialize ε_{t-1} to 0

    % --- Generate ARMA(1,1) Process ---
    for t = 2:T
        % Recurrence relation for ARMA(1,1)
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon_prev;
        epsilon_prev = epsilon(t); % Update ε_{t-1}
    end
end


 % Our parameters
c = 2;              
phi = 0.95;         
theta = 0.25;       
nu = 4;             
T = 800;            
y0 = 40;            
burn_in = 50;       


% compute the expected value
expected_value = c / (1 - phi);
disp(['Expected Value: ', num2str(expected_value)]);


%% Task 1b) I simulate the process


rng(42);  % Set seed for reproducibility 
series = ARMA_simulator(T, c, phi, theta, nu, y0);

% remove the brun in phase
y = series((burn_in + 1):end); % Final time series with 750 observations


% save for the later task that we still have the realization of the (task
% 5)
y_safe = y;

%% Task 1c) Plot the realization of the process

figure; % Create a new figure window
hold on; % Allow multiple plots on the same figure


% I plot the series
plot(1:(T - burn_in), y, 'LineWidth', 1.5, 'Color', [0 0.4470 0.7410]);

% I plot the expected value using a red line
yline(expected_value, '--r', 'LineWidth', 1.5, ...
    'LabelHorizontalAlignment', 'right', 'LabelVerticalAlignment', 'bottom');


title('Simulated ARMA(1,1) Process', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Time', 'FontSize', 12); % Label for x-axis
ylabel('Value', 'FontSize', 12); % Label for y-axis
grid on; % Enable grid for better visualization


xlim([1 T - burn_in]); 
% Set y-axis to include a margin around the minimum and maximum values
ylim([min(y) - 5, max(y) + 5]);


legend({'Simulated Process', 'Expected Value'}, 'Location', 'best', 'FontSize', 12);


set(gca, 'FontSize', 12, 'Box', 'on'); % Adjust axis properties
hold off; % Release the hold on the figure

%% Task 2 Dicky Fuller test and test statistic

%% Estimate parameters with OLS
yt = y(2:end); 



yt_lag = y(1:end-1); % Lagged variable (values shifted by one time step)


% define the matrix for X
X = [ones(length(yt_lag), 1), yt_lag];


% Calculate the OLS estimates using the formula: b = (X'X)^(-1)X'y
b = (X' * X) \ (X' * yt);


alpha_hat = b(1); % Estimated intercept (drift term)
rho_hat = b(2);   % Estimated AR coefficient (ρ)


y_pred = X * b;          % Predicted values based on OLS estimates
residuals = yt - y_pred; % Residuals (difference between actual and predicted values)


% Variance of residuals: σ² = (1 / (n - k)) * Σ(residuals²)
% Adjusted for 2 parameters (intercept and slope)
s_squared = (1 / (length(yt) - 2)) * sum(residuals .^ 2);


% Covariance matrix: Cov(b) = σ² * (X'X)^(-1)
Cov_b = s_squared * inv(X' * X);


% Standard errors are the square root of the diagonal elements of the covariance matrix
s_errors = sqrt(diag(Cov_b));


s_error_alpha = s_errors(1); % Standard error for intercept (α)
s_error_rho = s_errors(2);   % Standard error for AR coefficient (ρ)

% --- Display Results ---
disp('Estimated coefficients (βˆ):');
disp(['c (constant): ', num2str(b(1))]); % Display intercept estimate
disp(['φ (AR parameter): ', num2str(b(2))]); % Display AR coefficient estimate
disp('Standard Errors (s.e.):');
% standard erros 
disp(['s.e.(c): ', num2str(s_error_alpha)]); 
disp(['s.e.(φ): ', num2str(s_error_rho)]);   

%% Now lets calculate the test statistic that rho is one (Dicky Fuller test for this case)
phi_hat = b(2);       % estimate for φ
se_phi = s_errors(2);      % and standard error for φ


% t-statistic for testing H0: rho = 1
t_stat = (phi_hat - 1) / se_phi;

disp(['t-statistic: ', num2str(t_stat)]);


% Lets compare with critical value
critical_value = -2.86; %  5% critical value for my case (check ATS or Hamilton)
if t_stat < critical_value
    disp('We can reject the null hypothesis (H0: φ = 1).');
else
    disp('We fail to reject the null hypothesis (H0: φ = 1).');
end

%% Task 3

%% I create the function to calculate the likelihood contributions

function logL_contributions = ml_contributions(params, y)

    %  it should yield us a ((T-1)×1) column vector of log likelihood contributions

    % I extract the four important parameters
    c = params(1);
    phi = params(2);
    theta = params(3);
    nu = params(4);
    
    % I define the length of my series 
    T = length(y);
    epsilons = zeros(T, 1); % I store zeros for the innobations
    logL_contributions = zeros(T-1, 1); %The log contributions will be one less as the series
    
    % I compute the series of innovations ε_t (the previous epsilon is zero
    % as I defined my 0 vector of epsilons before 
    for t = 2:T
        epsilons(t) = y(t) - c - phi * y(t-1) - theta * epsilons(t-1);
    end

    % Here I drop ε1 (the first innovation)
    epsilons = epsilons(2:end);

    %index of epsilons is now shifted the first is e2 as a result I get e2-
    %eT
    
    % Compute log likelihood contributions
    for t = 1:(T-1)
        % I compute the log likelihood contribution for my t
        logL_contributions(t) = gammaln((nu + 1) / 2) ...
                                - gammaln(nu / 2) ...
                                - 0.5 * log(nu * pi) ...
                                - (nu + 1) / 2 * log(1 + (epsilons(t)^2 / nu));
    end
end

%% I create the function to calculate the loglikelihood
function total_logL = loglikelihood(params, y)

    % I call the ml_contributions function that I have defined before
    logL_contributions = ml_contributions(params, y);
    
    % I get the total conditional log likelihood value by adding the
    % contributions
    total_logL = sum(logL_contributions);
end



%% Task 3b

% We are going to use the simulated series from before
y;


%The two cases we face

% Case a) c = 2, φ = 0.95, θ = 0.25, ν = 4 (those created the series)
params_a = [2; 0.95; 0.25; 4];
total_logL_a = loglikelihood(params_a, y);
disp(['Total conditional log likelihood for case a): ', num2str(total_logL_a)]);

% Case b) c = 1.5, φ = 0.75, θ = 0.5, ν = 6
params_b = [1.5; 0.75; 0.5; 6];
total_logL_b = loglikelihood(params_b, y);
disp(['Total conditional log likelihood for case b): ', num2str(total_logL_b)]);

%% just check how many calculated contributions i get out of my series

counter = ml_contributions(params_a, y);
length(counter)
% yeah I get 749

%% Task 4

%% The negative log likelihood function
function negLogL = neg_loglikelihood(params, y)

% call the likelihood contributions
    logL_contributions = ml_contributions(params, y);
    
    % I put a negative infront of the sum to calculat the negative
    negLogL = -sum(logL_contributions);
end
    


%% Estimate the parameters

% we need the toolbox
addpath('CML/CML');
% I define the starting values
x0 = [1.5; 0.75; 0.5; 5]; % for [c0, φ0, θ0, ν0]

% Define the optimization like provided in the task :
% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);


% I define my algotith

[x, f, g, cov, retcode] = CML(@neg_loglikelihood,   ...  
                              @ml_contributions,    ...  
                              y,                    ...  
                              x0,                   ...  
                              1,                    ...  
                              1,                    ...  
                              options);             ...  


% 6) Print or inspect the results
disp('Optimized parameters = ');
disp(x);

disp('Minimum negative log-likelihood = ');
disp(f);

disp('Gradient at optimum = ');
disp(g);

disp('Covariance matrix of estimates = ');
disp(cov);

disp(['Exit flag from fminsearch = ', num2str(retcode)]);

%% Task 4.3

% Here I will compute the confidence intervalls 

paramEst = x;      % [c; phi; theta; nu]
covEst   = cov;    % covariance matrix (4x4)

% 1) Compute standard errors from diagonal of covariance matrix
SE = sqrt(diag(covEst));

% 2) Construct 95% confidence intervals (two-sided)
zVal = 1.96;  % Approx. for 95% CI from the standard normal
CI_lower = paramEst - zVal .* SE;
CI_upper = paramEst + zVal .* SE;

% Display parameter estimates, SE, and 95% CIs
paramNames = {'c','phi','theta','nu'};
fprintf('\nParameter Estimates with 95%% CIs:\n');
for i = 1:length(paramEst)
    fprintf('%s: Estimate = %.4f, SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, paramEst(i), SE(i), CI_lower(i), CI_upper(i));
end


%% Task 4.4 

% here I perform the t-test



%    (Two-sided test)
phiEst = paramEst(2);
phiSE  = SE(2);

% Test statistic:  (phiEst - 0.8) / SE(phiEst)
tStat = (phiEst - 0.8) / phiSE;

% Critical value at alpha=5% (two-sided) from standard normal is about +-1.96
critVal = 1.96;

fprintf('\nT-test for H0: phi = 0.8\n');
fprintf('   Estimate of phi = %.4f\n', phiEst);
fprintf('   Standard Error   = %.4f\n', phiSE);
fprintf('   Test statistic   = %.4f\n', tStat);

if abs(tStat) > critVal
    fprintf('   => Reject H0 at the 5%% level (|tStat| > 1.96).\n\n');
else
    fprintf('   => Fail to reject H0 at the 5%% level.\n\n');
end


%% Task 4.5 (two sided)


% Here I compute the two sided p value
pValue = 2 * (1 - normcdf(abs(tStat)));  % two-sided

fprintf('Test statistic for H0: phi = 0.8 is tStat = %.4f\n', tStat);
fprintf('Two-sided p-value = %.6f\n', pValue);

% Interpretation:
if pValue < 0.05
    fprintf('=> p-value < 0.05 => Reject H0 at 5%% level.\n\n');
else
    fprintf('=> p-value >= 0.05 => Fail to reject H0 at 5%% level.\n\n');
end


%% Redo task 4 but with an increased sample size which will be 50000
% Now lets do the same but with 50000 simulations


%% Task 4.6
%Simualte the process 50000 times
% we still include the burn in stuff so we loose 50 observations
% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 50000;            % Total length of the series
y0 = c/(1-phi);     % Starting value (expected value of the process)



% Simulate the ARMA(1,1) process that creates a series of 800 values
rng(200)

% We will get now a new series of 50000 observation length
series = ARMA_simulator(T, c, phi, theta, nu, y0);

% We rename it as y as we need it for the following tasks
y = series((burn_in+1):end);

%%
% Now first lets estimate the paramter values again

% Set a seed

rng(200)
% Same starting values
x0 = [1.5; 0.75; 0.5; 5]; % for [c0, φ0, θ0, ν0]

% Parametization is defined as the same
% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);



% Now lets call the CML function

[x, f, g, cov, retcode] = CML(@neg_loglikelihood,   ...  
                              @ml_contributions,    ...  
                              y,                    ...  
                              x0,                   ...  
                              1,                    ...  
                              1,                    ...  
                              options);             ...  


%  Print or inspect the results
disp('Optimized parameters = ');
disp(x);

disp('Minimum negative log-likelihood = ');
disp(f);

disp('Gradient at optimum = ');
disp(g);

disp('Covariance matrix of estimates = ');
disp(cov);

disp(['Exit flag from fminsearch = ', num2str(retcode)]);


%% Lets check the standard errors and confidence intervalls


% Now for this sample size I compute the confidence intervalls
paramEst = x;      % [c; phi; theta; nu]
covEst   = cov;    % covariance matrix (4x4)

% 1) Compute standard errors from diagonal of covariance matrix
SE = sqrt(diag(covEst));

% 2) Construct 95% confidence intervals (two-sided)
zVal = 1.96;  % Approx. for 95% CI from the standard normal
CI_lower = paramEst - zVal .* SE;
CI_upper = paramEst + zVal .* SE;

% Display parameter estimates, SE, and 95% CIs
paramNames = {'c','phi','theta','nu'};
fprintf('\nParameter Estimates with 95%% CIs:\n');
for i = 1:length(paramEst)
    fprintf('%s: Estimate = %.4f, SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, paramEst(i), SE(i), CI_lower(i), CI_upper(i));
end

%% Lets perform our two sided t test 

% 3) Perform t-test for H0: phi = 0.8 at 5% significance
%    (Two-sided test)
phiEst = paramEst(2);
phiSE  = SE(2);

% Test statistic:  (phiEst - 0.8) / SE(phiEst)
tStat = (phiEst - 0.8) / phiSE;

% Critical value at alpha=5% (two-sided) from standard normal is about +-1.96
critVal = 1.96;

fprintf('\nT-test for H0: phi = 0.8\n');
fprintf('   Estimate of phi = %.4f\n', phiEst);
fprintf('   Standard Error   = %.4f\n', phiSE);
fprintf('   Test statistic   = %.4f\n', tStat);

if abs(tStat) > critVal
    fprintf('   => Reject H0 at the 5%% level (|tStat| > 1.96).\n\n');
else
    fprintf('   => Fail to reject H0 at the 5%% level.\n\n');
end

%% What about the two sided t tets 



% -- Now compute the two-sided p-value using the standard normal distribution
pValue = 2 * (1 - normcdf(abs(tStat)));  % two-sided

fprintf('Test statistic for H0: phi = 0.8 is tStat = %.4f\n', tStat);
fprintf('Two-sided p-value = %.6f\n', pValue);

% Interpretation:
if pValue < 0.05
    fprintf('=> p-value < 0.05 => Reject H0 at 5%% level.\n\n');
else
    fprintf('=> p-value >= 0.05 => Fail to reject H0 at 5%% level.\n\n');
end

%% Task 5 Quasi Maximum Likelihood

%% Function that computes the quasi maximum likelihood contributions
function logL_contributions = qml_contributions(params, y)


    % Extract parameters
    c      = params(1);
    phi    = params(2);
    theta  = params(3);
    sigma2 = params(4);


    T = length(y);
    epsilons = zeros(T, 1);

    % Compute the series of innovations:
    %   epsilon_t = y_t - c - phi*y_{t-1} - theta*epsilon_{t-1}
    for t = 2:T
        epsilons(t) = y(t) - c - phi*y(t-1) - theta*epsilons(t-1);
    end

    % Drop the first innovation (t=1 has no "previous" y or epsilon)
    epsilons = epsilons(2:end); % => size (T-1)x1

    % Compute log-likelihood contributions for Gaussian:
    %   ln f(epsilon_t) = -0.5*ln(2*pi*sigma^2) - (epsilon_t^2)/(2*sigma^2)
    logL_contributions = -0.5 * log(2*pi*sigma2) ...
                         - 0.5 * (epsilons.^2 ./ sigma2);
end



%% Function that computes the complete likelihood
% qml_l
function total_logL = qml_l(params, y)

    logL_contributions = qml_contributions(params, y);

    % Sum over t = 2..T
    total_logL = -sum(logL_contributions);
end




%% Task 5.3

% We use the realization of the process from task 1
y = y_safe;

% Lets define a rang before to ensure reproducibility
rng(200)



x0 = [1.5; 0.75; 0.5; 1];  % as given ( the last one is the variance now)



% I set the correct options
options = optimset('Display','iter',...
                   'TolX',1e-40,...
                   'TolFun',1e-40,...
                   'MaxIter',1e10,...
                   'MaxFunEvals',100000);




% I use the toolbos
[x_qml, fval_qml, g_qml, cov_qml, retcode_qml] = ...
    CML(@qml_l,                       ...  
        @qml_contributions,           ...  
        y,                            ...  
        x0,                           ...  
        1,                            ...  
        3,                            ...  
        options);



% I disply my results
fprintf('QML results (Gaussian assumption) for ARMA(1,1):\n');
fprintf('Estimated parameters:\n');
disp(x_qml);         % [c; phi; theta; sigma^2]

fprintf('Negative Log-Likelihood at optimum = %.4f\n', fval_qml);
fprintf('Gradient at optimum:\n');
disp(g_qml);
fprintf('Covariance matrix (QML-based):\n');
disp(cov_qml);
fprintf('Exit flag: %d\n', retcode_qml);

%% Task 5.4

% --- Suppose your final results from the qCML estimation were stored as:
paramEstQML = x_qml;      % [c; phi; theta; nu]
covEstQML   = cov_qml;    % covariance matrix (4x4)

% 1) Compute standard errors from diagonal of covariance matrix
SE_qml = sqrt(diag(covEstQML));

% 2) Construct 95% confidence intervals (two-sided)
zVal = 1.96;  % Approx. for 95% CI from the standard normal
CI_lower_qml = paramEstQML - zVal .* SE_qml;
CI_upper_qml = paramEstQML + zVal .* SE_qml;

% Display parameter estimates, SE, and 95% CIs
paramNames = {'c','phi','theta','sigma^2'};
fprintf('\nParameter Estimates with 95%% CIs:\n');
for i = 1:length(paramEstQML)
    fprintf('%s: Estimate = %.4f, SE = %.4f, CI = [%.4f, %.4f]\n', ...
        paramNames{i}, paramEstQML(i), SE_qml(i), CI_lower_qml(i), CI_upper_qml(i));
end


%% Task 6

%% Task 6a)

% here I am gonna define the parameters
c = 2;              
phi = 0.95;         
theta = 0.25;       
nu = 4;             
T = 800;            
burn_in = 50;       
K = 2500;           
y0 = 40;            

% --- Initializations for storage (the matrices)
ML_estimates = zeros(K, 4);   % To store ML estimates (c, phi, theta, nu) (kx4)
QML_estimates = zeros(K, 4);  % To store QML estimates (c, phi, theta, sigma^2) (kx4)
ML_covariances = zeros(K, 4, 4); % To store ML covariance matrices
ML_standard_errors = zeros(K, 4); % To store ML standard errors (sqrt of diagonal of cov matrices)


% --- Starting values for estimation
x0_ML = [1.5; 0.75; 0.5; 5];  % Starting values for ML
x0_QML = [1.5; 0.75; 0.5; 1]; % Starting values for QML


% --- Optimization settings
options = optimset('Display', 'off', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);
algorithm = 1; % fminsearch
covPar_ML = 1;  % Hessian-based covariance for ML
covPar_QML = 3; % QML-based covariance for QML

rng(42)
% --- Loop over K realizations
for k = 1:K
    % --- Step 1: Simulate ARMA(1,1) series
    series = ARMA_simulator(T , c, phi, theta, nu, y0);
    y = series((burn_in + 1):end);  % Remove burn-in phase (should be 750)

    % --- Step 2: Estimate parameters with ML (t-distributed)
    [ML_params, ~, ~, ML_cov, ~] = CML(@neg_loglikelihood, ...
                                       @ml_contributions, ...
                                       y, x0_ML, algorithm, covPar_ML, options);
    ML_estimates(k, :) = ML_params';
    ML_covariances(k, :, :) = ML_cov;
    ML_standard_errors(k, :) = sqrt(diag(ML_cov))';

    % --- Step 3: Estimate parameters with QML (Gaussian)
    [QML_params, ~, ~, ~, ~] = CML(@qml_l                    , ...
                                   @qml_contributions        , ...
                                   y, x0_QML, algorithm, covPar_QML, options);
    QML_estimates(k, :) = QML_params';
end

% --- Output storage completed
disp('Simulation and estimation complete.');
disp('ML estimates, QML estimates, and ML standard errors stored.');


%% Task 6b
% Compute kernel density estimates and plot densities for c, phi, and theta
params = {'c', 'phi', 'theta'};  % Parameter labels
colors = {'b', 'r'};             % Colors for ML (blue) and QML (red)

figure;  % Create a figure to plot densities
for i = 1:3  % Loop over parameters (c=1, phi=2, theta=3)
    subplot(1, 3, i);  % Create subplots for side-by-side density plots
    
    % --- Compute KDE for ML estimates
    [ML_density, ML_x] = ksdensity(ML_estimates(:, i));
    
    % --- Compute KDE for QML estimates
    [QML_density, QML_x] = ksdensity(QML_estimates(:, i));
    
    % --- Plot the ML KDE
    plot(ML_x, ML_density, 'LineWidth', 2, 'Color', colors{1});
    hold on;  % Allow overlaying QML plot
    
    % --- Plot the QML KDE
    plot(QML_x, QML_density, 'LineWidth', 2, 'Color', colors{2});
    
    % --- Format the plot
    title(['Kernel Density of ', params{i}], 'FontSize', 12);
    xlabel(params{i}, 'FontSize', 12);
    ylabel('Density', 'FontSize', 12);
    legend({'ML', 'QML'}, 'Location', 'Best', 'FontSize', 10);
    grid on;
    hold off;  % End overlay
end

% --- Make the figure look professional
set(gcf, 'Color', 'w');  % Set background to white
sgtitle('Kernel Densities of ML and QML Estimates', 'FontSize', 14);


%% Task7
%% Calculate the coverage rates of the confidence intervall

% --- True parameter values
true_params = [2; 0.95; 0.25; 4];  % [c, phi, theta, nu]

% --- Number of ensembles
K = size(ML_estimates, 1);  % Should already be 2500 from Task 6.1

% --- Compute 95% Confidence Intervals for ML estimates
z_value = 1.96;  % For 95% CI
ML_CI_lower = ML_estimates - z_value .* ML_standard_errors;  % Lower bounds
ML_CI_upper = ML_estimates + z_value .* ML_standard_errors;  % Upper bounds

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

