function logL_contributions = arma_qml_log_likelihood_contributions(params, y)
    % Computes the log likelihood contributions for ARMA(1,1) with Gaussian innovations
    % INPUTS:
    %   params - A column vector of parameters [c; φ; θ; σ]
    %       c   - Constant term
    %       φ   - AR parameter
    %       θ   - MA parameter
    %       σ   - Standard deviation of Gaussian innovations
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   logL_contributions - A ((T-1)×1) column vector of log likelihood contributions

    % Extract parameters
    c = params(1);
    phi = params(2);
    theta = params(3);
    sigma = params(4);
    
    % Initialize variables
    T = length(y);
    epsilons = zeros(T, 1); % Innovations vector ε_t
    logL_contributions = zeros(T-1, 1); % Log likelihood contributions

    % Compute the series of innovations ε_t
    for t = 2:T
        epsilons(t) = y(t) - c - phi * y(t-1) - theta * epsilons(t-1);
    end

    % Drop ε1 (the first innovation)
    epsilons = epsilons(2:end);
    
    % Compute log likelihood contributions
    for t = 1:(T-1)
        logL_contributions(t) = -0.5 * log(2 * pi * sigma^2) - (epsilons(t)^2) / (2 * sigma^2);
    end
end
