%% function
function yt = simulate_ARMA_t(T, c, phi, theta, nu, y0)
    % Function to simulate an ARMA(1,1) process with t-distributed innovations
    % The model: yt = c + φyt-1 + εt + θεt-1, where ε ~ t(ν)

    % Preallocate arrays for yt and epsilon (innovations)
    yt = zeros(T, 1);  % Time series output
    epsilon = zeros(T, 1);  % Innovations (ε_t)
    
    % Generate t-distributed random errors (innovations)
    epsilon = trnd(nu, T, 1);  % Generate T samples from t(ν)
    
    % Initialize the process
    yt(1) = y0;  % for yt you just select the previous value y0 which is 0
    epsilon_prev = 0;  % Assume ε_{t-1} is 0 for the first step

    % Loop to generate the ARMA process
    for t = 2:T
        % ARMA(1,1) recursion formula
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon_prev;
        
        % Update ε_{t-1} for the next step
        epsilon_prev = epsilon(t);
    end
end

%%
%%Simualte the process
% Parameters
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Total length of the series
y0 = c/(1-phi);     % Starting value (expected value of the process)
burn_in = 50;       % Burn-in phase length

expected_value = c/(1-phi);
disp(expected_value)


% Simulate the ARMA(1,1) process that creates a series of 800 values
rng(200)
series= simulate_ARMA_t(T, c, phi, theta, nu, y0);



% Remove the first 50 observations
y = series((burn_in+1):end); % works as we have 750 length vector

%%
% Now the estimation part starts 

% First I choose the response variable
yt = y(2:end); % response variable

% moreobver the lagged value
yt_lag = y(1:end-1); % lagged variable

% Add intercept term to X
X = [ones(length(yt_lag), 1), yt_lag];

% Compute OLS estimates
b = (X' * X) \ (X' * yt);

% Extract alpha and rho
alpha_hat = b(1);
rho_hat = b(2);

% First lets get the predictions
y_pred = X* b;

% and then get the resiudals
residuals = yt - y_pred;

% Estimate sigma squared (variance of residuals)
s_squared = (1 / (length(yt) - 2)) * sum(residuals .^ 2); % Adjusted for 2 parameters (alpha and rho)

% Geth the covariance matrix
Cov_b = s_squared * inv(X' * X);



% Here we take for the standard error the squarred of diagonal elements of
% the variance
s_errors = sqrt(diag(Cov_b));

% The standard error for alpha
s_error_alpha = s_errors(1);
% The standard error rho
s_error_rho   = s_errors(2);



% Display the estimated coefficients
disp('Estimated coefficients (βˆ):');
disp(['c (constant): ', num2str(b(1))]);
disp(['φ (AR parameter): ', num2str(b(2))]);
disp('Standard Errors (s.e.):');
disp(['s.e.(c): ', num2str(s_errors(1))]); % indicates uncertainty in estimating the drift
disp(['s.e.(φ): ', num2str(s_errors(2))]); % indicates uncertainty in estimating the ar component

%% Now lets calculate the test statistic that rho is one
phi_hat = b(2);       % Estimated φ
se_phi = s_errors(2);      % Standard error of φ


% t-statistic for testing H0: rho = 1
t_stat = (phi_hat - 1) / se_phi;

disp(['t-statistic: ', num2str(t_stat)]);


% Compare with critical value
critical_value = -2.86; % Example: 5% critical value for drift case (check ATS or Hamilton)
if t_stat < critical_value
    disp('Reject the null hypothesis (H0: φ = 1). The series is stationary.');
else
    disp('Fail to reject the null hypothesis (H0: φ = 1). The series is non-stationary.');
end
