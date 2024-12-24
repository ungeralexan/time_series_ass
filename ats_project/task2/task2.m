%% Task2

%% Task 2 b
% the estimation of the series with OLS

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

% Display the results of my estimates
disp('Estimated coefficients (βˆ):');
disp(['c (constant): ', num2str(b(1))]);
disp(['φ (AR parameter): ', num2str(b(2))]);
disp('Standard Errors (s.e.):');
disp(['s.e.(c): ', num2str(s_errors(1))]); % indicates uncertainty in estimating the drift
disp(['s.e.(φ): ', num2str(s_errors(2))]); % indicates uncertainty in estimating the ar component



%% Now lets calculate the test statistic that rho is one (Dicky Fuller test for this case9
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