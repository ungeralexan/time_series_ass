function negLogL = arma_qml_log_likelihood(params, y)
    % Computes the negative total log likelihood for ARMA(1,1) with Gaussian innovations
    % INPUTS:
    %   params - A column vector of parameters [c; φ; θ; σ]
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   negLogL - The negative total log likelihood (scalar)

    % Call the log likelihood contributions function
    logL_contributions = arma_qml_log_likelihood_contributions(params, y);
    
    % Sum the log likelihood contributions and return the negative value
    negLogL = -sum(logL_contributions);
end
