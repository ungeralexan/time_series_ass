% Load the ARMA(1,1) realization from Task 2.1
% Assume 'yt' is the time series data generated in Task 2.1
y = yt; % Replace 'yt' with the actual variable containing your series

% Case a) c = 2, φ = 0.95, θ = 0.25, ν = 4
params_a = [2; 0.95; 0.25; 4];
total_logL_a = compute_conditional_log_likelihood(params_a, y);
disp(['Total conditional log likelihood for case a): ', num2str(total_logL_a)]);

% Case b) c = 1.5, φ = 0.75, θ = 0.5, ν = 6
params_b = [1.5; 0.75; 0.5; 6];
total_logL_b = compute_conditional_log_likelihood(params_b, y);
disp(['Total conditional log likelihood for case b): ', num2str(total_logL_b)]);
