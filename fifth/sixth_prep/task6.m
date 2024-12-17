%% Task 6a
% Set parameters for simulation
K = 2500;                % Number of ensembles
T = 800;                 % Total sample size
burn_in = 50;            % Burn-in phase
params_true = [2; 0.95; 0.25; 4]; % True parameters: [c, φ, θ, ν]
sigma_true = 1;          % Standard deviation for QML assumption

% Initialize storage matrices
ML_estimates = zeros(K, 4);    % Store ML estimates: [c, φ, θ, ν]
QML_estimates = zeros(K, 4);   % Store QML estimates: [c, φ, θ, σ]
ML_std_errors = zeros(K, 4);   % Standard errors for ML
QML_std_errors = zeros(K, 4);  % Standard errors for QML

% Optimization settings
options = optimset('Display', 'off', 'TolX', 1e-6, 'TolFun', 1e-6, ...
                   'MaxIter', 10000, 'MaxFunEvals', 10000);

% Simulate K ensembles and estimate parameters
for k = 1:K
    % Step 1: Simulate ARMA(1,1) process
    y = simulate_ARMA_t(T, params_true(1), params_true(2), params_true(3), params_true(4), 0);
    y = y((burn_in+1):end); % Drop burn-in phase
    
    % Step 2: ML Estimation (t-distributed innovations)
    x0_ML = [1.5; 0.75; 0.5; 5]; % Initial guesses
    [x_ML, ~, ~, cov_ML, ~] = CML(@arma_log_likelihood, @arma_conditional_log_likelihood, y, x0_ML, 1, 1, options);
    
    % Store ML results
    ML_estimates(k, :) = x_ML';             % Parameter estimates
    ML_std_errors(k, :) = sqrt(diag(cov_ML))'; % Standard errors
    
    % Step 3: QML Estimation (Gaussian innovations)
    x0_QML = [1.5; 0.75; 0.5; sigma_true]; % Initial guesses
    [x_QML, ~, ~, cov_QML, ~] = CML(@arma_qml_log_likelihood, @arma_qml_log_likelihood_contributions, y, x0_QML, 1, 3, options);
    
    % Store QML results
    QML_estimates(k, :) = x_QML';             % Parameter estimates
    QML_std_errors(k, :) = sqrt(diag(cov_QML))'; % Standard errors
end

% Save results
save('arma_results.mat', 'ML_estimates', 'QML_estimates', 'ML_std_errors', 'QML_std_errors');
disp('Simulation complete. Results stored.');

%% Task 6b
% Load results
load('arma_results.mat'); % ML_estimates, QML_estimates

% Extract parameters for c, φ, and θ
c_ML = ML_estimates(:, 1);
phi_ML = ML_estimates(:, 2);
theta_ML = ML_estimates(:, 3);

c_QML = QML_estimates(:, 1);
phi_QML = QML_estimates(:, 2);
theta_QML = QML_estimates(:, 3);

% Set up the figure
figure;

% Plot kernel densities for c (constant)
subplot(3, 1, 1);
[f_c_ML, xi_c_ML] = ksdensity(c_ML);
[f_c_QML, xi_c_QML] = ksdensity(c_QML);

plot(xi_c_ML, f_c_ML, 'LineWidth', 1.5, 'DisplayName', 'ML Estimate'); hold on;
plot(xi_c_QML, f_c_QML, '--', 'LineWidth', 1.5, 'DisplayName', 'QML Estimate');
title('Kernel Density of c (Constant Term)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('c'); ylabel('Density');
legend('show', 'Location', 'Best');
grid on; hold off;

% Plot kernel densities for φ (AR parameter)
subplot(3, 1, 2);
[f_phi_ML, xi_phi_ML] = ksdensity(phi_ML);
[f_phi_QML, xi_phi_QML] = ksdensity(phi_QML);

plot(xi_phi_ML, f_phi_ML, 'LineWidth', 1.5, 'DisplayName', 'ML Estimate'); hold on;
plot(xi_phi_QML, f_phi_QML, '--', 'LineWidth', 1.5, 'DisplayName', 'QML Estimate');
title('Kernel Density of \phi (AR Parameter)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('\phi'); ylabel('Density');
legend('show', 'Location', 'Best');
grid on; hold off;

% Plot kernel densities for θ (MA parameter)
subplot(3, 1, 3);
[f_theta_ML, xi_theta_ML] = ksdensity(theta_ML);
[f_theta_QML, xi_theta_QML] = ksdensity(theta_QML);

plot(xi_theta_ML, f_theta_ML, 'LineWidth', 1.5, 'DisplayName', 'ML Estimate'); hold on;
plot(xi_theta_QML, f_theta_QML, '--', 'LineWidth', 1.5, 'DisplayName', 'QML Estimate');
title('Kernel Density of \theta (MA Parameter)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('\theta'); ylabel('Density');
legend('show', 'Location', 'Best');
grid on; hold off;

% Adjust figure
sgtitle('Kernel Densities for ML and QML Estimates', 'FontSize', 14, 'FontWeight', 'bold');
