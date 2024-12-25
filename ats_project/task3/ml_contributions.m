function logL_contributions = ml_contributions(params, y)
    % ARMA(1,1) Conditional Log Likelihood Function with t-distributed Innovations
    %
    % INPUTS:
    %   params - A column vector of parameters [c; φ; θ; ν]
    %       c   - Constant term in the ARMA process
    %       φ   - AR parameter
    %       θ   - MA parameter
    %       ν   - Degrees of freedom for the t-distribution
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   logL_contributions - A ((T-1)×1) column vector of log likelihood contributions

    % Extract parameters
    c = params(1);
    phi = params(2);
    theta = params(3);
    nu = params(4);
    
    % Initialize variables
    T = length(y);
    epsilons = zeros(T, 1); % Innovations vector (ε_t)
    logL_contributions = zeros(T-1, 1); % Log likelihood contributions
    
    % Compute the series of innovations ε_t
    for t = 2:T
        epsilons(t) = y(t) - c - phi * y(t-1) - theta * epsilons(t-1);
    end

    % Drop ε1 (the first innovation)
    epsilons = epsilons(2:end);
    
    % Compute log likelihood contributions
    for t = 1:(T-1)
        % Compute the log likelihood contribution for each t
        logL_contributions(t) = gammaln((nu + 1) / 2) ...
                                - gammaln(nu / 2) ...
                                - 0.5 * log(nu * pi) ...
                                - (nu + 1) / 2 * log(1 + (epsilons(t)^2 / nu));
    end
end