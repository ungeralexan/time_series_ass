function negLogL = arma_log_likelihood(params, y)
    % Computes the negative total conditional log likelihood for ARMA(1,1)
    % INPUTS:
    %   params - A column vector of parameters [c; φ; θ; ν]
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   negLogL - The negative total log likelihood (scalar)
    
    % Call the conditional log likelihood contributions function
    logL_contributions = arma_conditional_log_likelihood(params, y);
    
    % Compute the total log likelihood and return its negative
    negLogL = -sum(logL_contributions);
end
