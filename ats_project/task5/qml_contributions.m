function logL_contributions = qml_contributions(params, y)
    % ml_contributions_gaussian:
    %   Computes the per-observation log-likelihood contributions for an
    %   ARMA(1,1) model with Gaussian (N(0, sigma^2)) innovations.
    %
    % Model:
    %   y_t = c + phi * y_{t-1} + epsilon_t + theta * epsilon_{t-1},
    %   where  epsilon_t ~ N(0, sigma^2).
    %
    % INPUTS:
    %   params - A 4x1 column vector [c; phi; theta; sigma2],
    %            where sigma2 is the variance (not the st.dev.)
    %   y      - A column vector of time series data [y1; y2; ...; yT].
    %
    % OUTPUT:
    %   logL_contributions - A (T-1)x1 vector of per-observation log-likelihoods.

    % Extract parameters
    c      = params(1);
    phi    = params(2);
    theta  = params(3);
    sigma2 = params(4);


    T = length(y);
    epsilons = zeros(T, 1);

    % Compute the series of innovations:
    %   epsilon_t = y_t - c - phi*y_{t-1} - theta*epsilon_{t-1}
    for t = 2:T
        epsilons(t) = y(t) - c - phi*y(t-1) - theta*epsilons(t-1);
    end

    % Drop the first innovation (t=1 has no "previous" y or epsilon)
    epsilons = epsilons(2:end); % => size (T-1)x1

    % Compute log-likelihood contributions for Gaussian:
    %   ln f(epsilon_t) = -0.5*ln(2*pi*sigma^2) - (epsilon_t^2)/(2*sigma^2)
    logL_contributions = -0.5 * log(2*pi*sigma2) ...
                         - 0.5 * (epsilons.^2 ./ sigma2);
end
