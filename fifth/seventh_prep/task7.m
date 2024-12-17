% True parameter values (used for simulation)
true_params = [2; 0.95; 0.25; 4]; % [c, φ, θ, ν]

% Extract ML estimates and standard errors
c_ML = ML_estimates(:, 1);
phi_ML = ML_estimates(:, 2);
theta_ML = ML_estimates(:, 3);
nu_ML = ML_estimates(:, 4);

se_c = ML_std_errors(:, 1);
se_phi = ML_std_errors(:, 2);
se_theta = ML_std_errors(:, 3);
se_nu = ML_std_errors(:, 4);

% Compute 95% confidence intervals
lower_c = c_ML - 1.96 * se_c;
upper_c = c_ML + 1.96 * se_c;

lower_phi = phi_ML - 1.96 * se_phi;
upper_phi = phi_ML + 1.96 * se_phi;

lower_theta = theta_ML - 1.96 * se_theta;
upper_theta = theta_ML + 1.96 * se_theta;

lower_nu = nu_ML - 1.96 * se_nu;
upper_nu = nu_ML + 1.96 * se_nu;

% Test how often true parameters lie within the confidence intervals
c_in_CI = sum((true_params(1) >= lower_c) & (true_params(1) <= upper_c));
phi_in_CI = sum((true_params(2) >= lower_phi) & (true_params(2) <= upper_phi));
theta_in_CI = sum((true_params(3) >= lower_theta) & (true_params(3) <= upper_theta));
nu_in_CI = sum((true_params(4) >= lower_nu) & (true_params(4) <= upper_nu));

% Compute percentages
c_coverage = (c_in_CI / 2500) * 100;
phi_coverage = (phi_in_CI / 2500) * 100;
theta_coverage = (theta_in_CI / 2500) * 100;
nu_coverage = (nu_in_CI / 2500) * 100;

% Display results
disp('95% Confidence Interval Coverage Results:');
disp('-----------------------------------------');
fprintf('Parameter c (true value = %.2f): %.2f%% of intervals include the true value\n', true_params(1), c_coverage);
fprintf('Parameter φ (true value = %.2f): %.2f%% of intervals include the true value\n', true_params(2), phi_coverage);
fprintf('Parameter θ (true value = %.2f): %.2f%% of intervals include the true value\n', true_params(3), theta_coverage);
fprintf('Parameter ν (true value = %.2f): %.2f%% of intervals include the true value\n', true_params(4), nu_coverage);
disp('-----------------------------------------');
