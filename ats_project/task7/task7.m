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
