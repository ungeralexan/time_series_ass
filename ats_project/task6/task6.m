%% Task6

% --- Parameters from Task 1
c = 2;              % Constant term
phi = 0.95;         % AR parameter
theta = 0.25;       % MA parameter
nu = 4;             % Degrees of freedom for t-distribution
T = 800;            % Series length
burn_in = 50;       % Burn-in phase
K = 2500;           % Number of ensembles
y0 = 40;            % Initial value

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
