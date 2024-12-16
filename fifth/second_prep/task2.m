%% Task 2b
% Load the processed time series data (after burn-in phase)
% Assume yt contains the processed time series after Task 1.3.

% Construct the dependent variable y (starting from the second observation)
y = yt(2:end);

% Construct the regressor matrix X (constant and lagged values of yt)
X = [ones(length(yt)-1, 1), yt(1:end-1)];

% Compute the OLS estimator
beta_hat = (X' * X) \ (X' * y);

% Display the estimated coefficients
disp('Estimated coefficients (βˆ):');
disp(['c (constant): ', num2str(beta_hat(1))]);
disp(['φ (AR parameter): ', num2str(beta_hat(2))]);


%% Task 2c) construct the standard errors
% Step 1: Residual Variance (s^2)
T = length(y);  % Number of observations
K = size(X, 2); % Number of regressors (columns in X)
residuals = y - X * beta_hat; % Compute residuals
s2 = (1 / (T - K)) * sum(residuals .^ 2); % Residual variance

% Step 2: Compute (X'X)^(-1)
XX_inv = inv(X' * X);

% Step 3: Compute standard errors
std_errors = sqrt(diag(s2 * XX_inv));

% Display results
disp('Standard Errors (s.e.):');
disp(['s.e.(c): ', num2str(std_errors(1))]); % indicates uncertainty in estimating the drift
disp(['s.e.(φ): ', num2str(std_errors(2))]); % indicates uncertainty in estimating the ar component

%% Dicky Fuller test statistic results

% Assuming `beta_hat` contains estimated coefficients
% beta_hat(1): constant c, beta_hat(2): φ (AR parameter)

phi_hat = beta_hat(2);       % Estimated φ
se_phi = std_errors(2);      % Standard error of φ

% Compute the t-statistic
t_phi = (phi_hat - 1) / se_phi;

% Display results
disp('Dickey-Fuller Test Results:');
disp(['Estimated φ: ', num2str(phi_hat)]);
disp(['Standard error: ', num2str(se_phi)]);
disp(['t-statistic: ', num2str(t_phi)]);

% Compare with critical value
critical_value = -2.86; % Example: 5% critical value for drift case (check ATS or Hamilton)
if t_phi < critical_value
    disp('Reject the null hypothesis (H0: φ = 1). The series is stationary.');
else
    disp('Fail to reject the null hypothesis (H0: φ = 1). The series is non-stationary.');
end

