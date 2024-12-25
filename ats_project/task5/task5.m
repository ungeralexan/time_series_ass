%% Task 5.3
% we have the same realization as in task 1 (we load it here)
y = series;


% And again define our starting values as before 
% ----------------------------------------------------
% 3) Define starting values
%    [ c_0, phi_0, theta_0, sigma^2_0 ]
% ----------------------------------------------------
x0 = [1.5; 0.75; 0.5; 1];  % as given ( the last one is the variance now)

% ----------------------------------------------------
% 4) Set the optimization options
%    (similar to previous tasks)
% ----------------------------------------------------
options = optimset('Display','iter',...
                   'TolX',1e-40,...
                   'TolFun',1e-40,...
                   'MaxIter',1e10,...
                   'MaxFunEvals',100000);


% ----------------------------------------------------
% 5) Choose the algorithm and the covariance method:
%    - algorithm = 1 => fminsearch
%    - covPar = 3  => QML-based covariance matrix
% ----------------------------------------------------
%algorithm = 1;  % (1: fminsearch)
%covPar    = 3;  % (3: QML-based covariance in the CML function)

% ----------------------------------------------------
% 6) Run the CML toolbox
%    We pass the negative log-likelihood and the contributions
% ----------------------------------------------------
[x_qml, fval_qml, g_qml, cov_qml, retcode_qml] = ...
    CML(@qml_l,                       ...  % fun1
        @qml_contributions,           ...  % fun2
        y,                            ...  % dataset
        x0,                           ...  % starting values
        1,                            ...  % which optimization
        3,                            ...  % QML-based covariance
        options);

% ----------------------------------------------------
% 7) Inspect the results
% ----------------------------------------------------
fprintf('QML results (Gaussian assumption) for ARMA(1,1):\n');
fprintf('Estimated parameters:\n');
disp(x_qml);         % [c; phi; theta; sigma^2]

fprintf('Negative Log-Likelihood at optimum = %.4f\n', fval_qml);
fprintf('Gradient at optimum:\n');
disp(g_qml);
fprintf('Covariance matrix (QML-based):\n');
disp(cov_qml);
fprintf('Exit flag: %d\n', retcode_qml);