function total_logL = qml_l(params, y)
    % loglikelihood_gaussian:
    %   Sums the per-observation log-likelihood contributions for an
    %   ARMA(1,1) model with Gaussian innovations.
    %
    % INPUTS:
    %   params - A 4x1 column vector [c; phi; theta; sigma2]
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   total_logL - The total conditional log-likelihood (scalar).

    % Get the contributions from the helper function
    logL_contributions = qml_contributions(params, y);

    % Sum over t = 2..T
    total_logL = -sum(logL_contributions);
end
