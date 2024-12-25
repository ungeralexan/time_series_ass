function total_logL = loglikelihood(params, y)
    % Computes the total conditional log likelihood for an ARMA(1,1) process
    % with t-distributed innovations by calling the arma_conditional_log_likelihood function.
    %
    % INPUTS:
    %   params - A column vector of parameters [c; φ; θ; ν]
    %   y      - A column vector of time series data [y1; y2; ...; yT]
    %
    % OUTPUT:
    %   total_logL - The total conditional log likelihood value (scalar)
    
    % Call the arma_conditional_log_likelihood function
    logL_contributions = ml_contributions(params, y);
    
    % Compute the total log likelihood by summing the contributions
    total_logL = sum(logL_contributions);
end