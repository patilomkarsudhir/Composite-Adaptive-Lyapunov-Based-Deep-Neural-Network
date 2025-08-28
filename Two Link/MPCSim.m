function [e, u_out, time, X, Xd] = MPCSim(n, step_size, simtime, X_init, alpha_1, prediction_horizon, Q_ei, Q_ri, R, u_bound)
% This function simulates a system using Model Predictive Control (MPC) with fmincon for trajectory tracking.
% The MPC is designed to track a time-varying reference trajectory with noisy state measurements.

% Initialize parameters and states
H = prediction_horizon;
time_length = round(simtime / step_size); % Changed from division to round to ensure an integer value for time_length
X = X_init;
time = zeros(1, time_length);
u_out = zeros(n, time_length);
Xd = zeros(size(X_init, 1), time_length);
e = zeros(n, time_length);

% Measurement noise parameters based on Signal-to-Noise Ratio (SNR)
SNR = 50; % Signal-to-Noise Ratio in dB

for i = 1:time_length
    % Time Calculation
    t = (i - 1) * step_size;
    time(i) = t;

    % System State Extraction
    Xi = X(:, i);
    xi = X(1:n, i);                % State (positions)
    xi_dot = X(n+1:2*n, i);        % State (velocities)
    
    % Calculate noise standard deviation based on SNR
    signal_power = mean(xi.^2) + 1e-6;
    noise_power = signal_power / (10^(SNR / 10));
    noise_std_dev = sqrt(noise_power * step_size);
    
    % Add measurement noise to the state
    noisy_xi = xi + noise_std_dev * randn(size(xi));
    noisy_xi_dot = xi_dot + noise_std_dev * randn(size(xi_dot));
    
    % Desired Trajectory and Errors
    A = 2;
    B = 0.5;
    f1 = 0.5;
    
    % Desired Trajectory
    xdi = 0.5 * 0.5 * exp(-sin(t)) * [sin(t); cos(t)];  % Desired Trajectory
    xdi_dot = 0.5 * (0.5 * exp(-sin(t)) * [cos(t); -sin(t)] - 0.5 * exp(-sin(t)) * cos(t) * [sin(t); cos(t)]);
    
    Xdi = [xdi; xdi_dot];
    Xd(:, i) = Xdi;
    
    % Noisy Errors for Control Development
    noisy_ei = noisy_xi - xdi;
    noisy_ei_dot = noisy_xi_dot - xdi_dot;
    
    % Actual Errors for Output
    ei = xi - xdi;
    ei_dot = xi_dot - xdi_dot;
    e(:, i) = ei;
    
    % Model Predictive Control (MPC) using fmincon
    % Predictive horizon states and controls
    
    % Define the initial guess for the control sequence over the entire horizon
    if i > 1
        U = repmat(u_out(:, i-1), H, 1); % Use the previous control action as the initial guess for the entire horizon
    else
        U = repmat(zeros(n, 1), H, 1); % Use zeros as the initial guess for the first iteration
    end
    
    % Expand the bounds to cover the entire prediction horizon
    lower_bounds = repmat(-u_bound, H, 1);
    upper_bounds = repmat(u_bound, H, 1);

    step_size_pred = step_size; % Define step_size_pred to avoid undefined variable
    cost_fun = @(U) compute_horizon_cost(U, noisy_xi, noisy_xi_dot, t, A, B, f1, Q_ei, Q_ri, R, alpha_1, step_size_pred, H);
    
    % Solve optimization problem using fmincon
    options = optimoptions('fmincon', 'MaxIterations', 500, 'Display', 'off', 'StepTolerance', 1e-6, 'OptimalityTolerance', 1e-6);
    [U_opt, fval, exitflag] = fmincon(cost_fun, U, [], [], [], [], lower_bounds, upper_bounds, [], options);
    
    % Add error handling for optimization failure
    if exitflag <= 0
        warning('Optimization did not converge, using previous control input.');
        if i > 1
            U_opt = u_out(:, i-1);
        else
            U_opt = zeros(n, 1);
        end
    end
    
    % Use only the first control action from the optimal sequence
    u = U_opt(1:n);
    u_body = u;
    
    % System State Update using Euler Integration
    [f, g] = dynamics(Xi); % Removed unused Jacobian J to improve computational efficiency
    Xdot = [xi_dot; f] + [zeros(n, 1); g * u];
    
    if i < time_length % Ensure array bounds are not exceeded
        X(:, i + 1) = Xi + step_size * Xdot;
    end
    u_out(:, i) = u_body;
end
end

function cost = compute_horizon_cost(U, xi, xi_dot, t, A, B, f1, Q_ei, Q_ri, R, alpha_1, step_size_pred, H)
    cost = 0;
    xi_pred = xi;
    xi_dot_pred = xi_dot;

    for h = 1:H
        % Time update for reference trajectory
        t_h = t + (h - 1) * step_size_pred;
        
        % Desired trajectory at time t_h
        xdi_h = 0.5 * 0.5 * exp(-sin(t_h)) * [sin(t_h); cos(t_h)];  % Desired Trajectory
        xdi_dot_h = 0.5 * (0.5 * exp(-sin(t_h)) * [cos(t_h); -sin(t_h)] - 0.5 * exp(-sin(t_h)) * cos(t_h) * [sin(t_h); cos(t_h)]);
        
        % Extract control input for the current time step
        u_h = U((h-1)*length(xi) + 1 : h*length(xi));

        % Update predicted states
        [f, g] = dynamics([xi_pred; xi_dot_pred]);
        xi_dot_pred = xi_dot_pred + step_size_pred * (f + g * u_h);
        xi_pred = xi_pred + step_size_pred * xi_dot_pred;

        % Calculate predicted error
        ei_pred = xi_pred - xdi_h;
        ri_pred = xi_dot_pred - xdi_dot_h + alpha_1 * ei_pred;

        % Accumulate cost
        cost = cost + (ei_pred' * Q_ei * ei_pred) + (u_h' * R * u_h) + (ri_pred' * Q_ri * ri_pred);
    end
end