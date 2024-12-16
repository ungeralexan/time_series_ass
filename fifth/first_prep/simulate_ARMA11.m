function yt = simulate_ARMA11(T, c, phi, theta, nu, y0)
    % T    - Sample size (number of time steps)
    % c    - Constant term
    % phi  - AR parameter
    % theta - MA parameter
    % nu   - Degrees of freedom for the t-distribution
    % y0   - Starting value for y(0)
    %
    % Output:
    % yt   - Simulated ARMA(1,1) process
    
    % Pre-allocate arrays for speed
    yt = zeros(T, 1);  % Time series values
    epsilon = zeros(T, 1);  % Innovations

    % Set the initial value of yt
    yt(1) = y0;

    % Generate the t-distributed innovations
    epsilon(1) = trnd(nu);  % First innovation
    for t = 2:T
        epsilon(t) = trnd(nu);  % Draw from t-distribution
    end

    % Generate the ARMA(1,1) process
    for t = 2:T
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon(t-1);
    end
end
