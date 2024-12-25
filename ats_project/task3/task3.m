%% Task3

%% I would simulate my likelihood contributions with the whole series

% Therefore I define the y as new series with 800 obs
y = series;

% Case a) c = 2, φ = 0.95, θ = 0.25, ν = 4 (those created the series)
params_a = [2; 0.95; 0.25; 4];
total_logL_a = loglikelihood(params_a, y);
disp(['Total conditional log likelihood for case a): ', num2str(total_logL_a)]);

% Case b) c = 1.5, φ = 0.75, θ = 0.5, ν = 6
params_b = [1.5; 0.75; 0.5; 6];
total_logL_b = loglikelihood(params_b, y);
disp(['Total conditional log likelihood for case b): ', num2str(total_logL_b)]);
