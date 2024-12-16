function yt = simulate_ARMA_t(T, c, phi, theta, nu, y0)
    % Function to simulate an ARMA(1,1) process with t-distributed innovations
    % The model: yt = c + φyt-1 + εt + θεt-1, where ε ~ t(ν)

    % Preallocate arrays for yt and epsilon (innovations)
    yt = zeros(T, 1);  % Time series output
    epsilon = zeros(T, 1);  % Innovations (ε_t)
    
    % Generate t-distributed random errors (innovations)
    epsilon = trnd(nu, T, 1);  % Generate T samples from t(ν)
    
    % Initialize the process
    yt(1) = y0;  % for yt you just select the previous value y0 which is 0
    epsilon_prev = 0;  % Assume ε_{t-1} is 0 for the first step

    % Loop to generate the ARMA process
    for t = 2:T
        % ARMA(1,1) recursion formula
        yt(t) = c + phi * yt(t-1) + epsilon(t) + theta * epsilon_prev;
        
        % Update ε_{t-1} for the next step
        epsilon_prev = epsilon(t);
    end
end
