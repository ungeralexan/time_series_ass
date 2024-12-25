%% Task 4

%% Define the starting values 

% This is stored as a column vector
x0 = [1.5; 0.75; 0.5; 5]; % for [c0, φ0, θ0, ν0]

% Define the optimization like provided in the task :
% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% Other important functions I call (after I constructed them)
% 1) ml_contributions is the function that calculates the conditional
% likelihoods

% 2) neg_loglikelihood is the function that calculates the negative
% likelihood

%Important for the toolbox i will specify
% algorithm = 1 means it uses fminsearch (derivative-free).
% covPar = 1 means you want the Hessian-based covariance matrix.

% Now lets call the CML function

[x, f, g, cov, retcode] = CML(@neg_loglikelihood,   ...  % fun1
                              @ml_contributions,    ...  % fun2
                              y,                    ...  % dataset
                              x0,                   ...  % starting values
                              1,                    ...  % which optimization
                              1,                    ...  % covariance choice
                              options);             ...  % optimization options


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

% --- Suppose your final results from the CML estimation were stored as:
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


%% Task 4.5 (two sided)
% -- We assume you have already computed:
% phiEst = paramEst(2);   % your estimated phi
% phiSE  = SE(2);         % standard error of phi
% tStat  = (phiEst - 0.8) / phiSE;

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


%%
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
y = series;


%%
% Now first lets estimate the paramter values again




% Same starting values
x0 = [1.5; 0.75; 0.5; 5]; % for [c0, φ0, θ0, ν0]

% Parametization is defined as the same
% Optimization settings
options = optimset('Display', 'iter', 'TolX', 1e-40, 'TolFun', 1e-40, ...
                   'MaxIter', 1e10, 'MaxFunEvals', 100000);

% Other important functions I call (after I constructed them)
% 1) ml_contributions is the function that calculates the conditional
% likelihoods

% 2) neg_loglikelihood is the function that calculates the negative
% likelihood

%Important for the toolbox i will specify
% algorithm = 1 means it uses fminsearch (derivative-free).
% covPar = 1 means you want the Hessian-based covariance matrix.

% Now lets call the CML function

[x, f, g, cov, retcode] = CML(@neg_loglikelihood,   ...  % fun1
                              @ml_contributions,    ...  % fun2
                              y,                    ...  % dataset
                              x0,                   ...  % starting values
                              1,                    ...  % which optimization
                              1,                    ...  % covariance choice
                              options);             ...  % optimization options


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


% --- Suppose your final results from the CML estimation were stored as:
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

% -- We assume you have already computed:
% phiEst = paramEst(2);   % your estimated phi
% phiSE  = SE(2);         % standard error of phi
% tStat  = (phiEst - 0.8) / phiSE;

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
