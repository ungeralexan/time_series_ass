% This is the redone script from task 3.2
% I just use the y now I have simualted and have for y = 750 obs (we
% substracted the burned it values of the series)

% Case a) c = 2, φ = 0.95, θ = 0.25, ν = 4
params_a = [2; 0.95; 0.25; 4];
total_logL_a = compute_conditional_log_likelihood(params_a, y);
disp(['Total conditional log likelihood for case a): ', num2str(total_logL_a)]);

% will yield a value of (-1.2593)

% Case b) c = 1.5, φ = 0.75, θ = 0.5, ν = 6
params_b = [1.5; 0.75; 0.5; 6];
total_logL_b = compute_conditional_log_likelihood(params_b, y);
disp(['Total conditional log likelihood for case b): ', num2str(total_logL_b)]);

% will yield a value of (-5.5622) 


% Conclu : We see that the loglikelihood is significantly worse when you
% move away from the true parameters 
%%
% Questions ye:

% 1) Do I take all the observations or only the 750 with the deleted ones ?
