function [e, u_out, time, X, Xd] = MPCSimIF(n, step_size, simtime, X_init, kr, alpha_1, intervals, prediction_horizon, Q_ei, Q_ri, R, u_bound)
% This function simulates a system using Model Predictive Control (MPC) with fmincon for trajectory tracking.
% The MPC is designed to track a time-varying reference trajectory.

% Initialize parameters and states
time_length = simtime / step_size;
X = X_init;
time = zeros(1, time_length);
u_out = zeros(n, time_length);
Xd = zeros(size(X_init, 1), time_length);
e = zeros(n, time_length);

for i = 1:time_length
    % Time Calculation
    t = (i - 1) * step_size;
    time(i) = t;
    
    % Determine if we are within specified intervals to decide control mode
    flag = all(~(t >= intervals(:, 1) & t <= intervals(:, 2)));
    if flag
        sigma = 1;
    else
        sigma = 2;
    end
    
    % System State Extraction
    Xi = X(:, i);
    xi = X(1:n, i);                % State (positions)
    xi_dot = X(n+1:2*n, i);        % State (velocities)
    
    % Desired Trajectory and Errors
    A = 2;
    B = 0.5;
    f1 = 0.5;
    
    xdi = [A*cos(f1*t); A*sin(f1*t); B*f1*t; 0; 0; -0.5*B*f1*t]; % Desired trajectory
    xdi_dot = [-A*f1*sin(f1*t); A*f1*cos(f1*t); B*f1; 0; 0; -0.5*B*f1];
    xdi_ddot = [-A*f1^2*cos(f1*t); -A*f1^2*sin(f1*t); 0; 0; 0; 0];
    
    Xdi = [xdi; xdi_dot];
    Xd(:, i) = Xdi;
    ei = xi - xdi;
    ei_dot = xi_dot - xdi_dot;
    e(:, i) = ei;
    
    % Model Predictive Control (MPC) using fmincon
    if sigma == 1
        % Full Dynamics Known Mode
        % Predictive horizon states and controls
        H = prediction_horizon;
        step_size_pred = step_size / H; % Reduced step size for improved prediction accuracy
        
        % Define the initial guess for the control sequence over the entire horizon
        if i > 1
            U = repmat(u_out(:, i-1), H, 1); % Use the previous control action as the initial guess for the entire horizon
        else
            U = repmat(zeros(n, 1), H, 1); % Use zeros as the initial guess for the first iteration
        end
        
        % Expand the bounds to cover the entire prediction horizon
        lower_bounds = repmat(-u_bound, H, 1);
        upper_bounds = repmat(u_bound, H, 1);
        
        % Define cost function for the entire horizon
        cost_fun = @(U) compute_horizon_cost(U, xi, xi_dot, t, A, B, f1, Q_ei, Q_ri, R, alpha_1, step_size_pred, H);
        
        % Solve optimization problem using fmincon
        options = optimoptions('fmincon', 'MaxIterations', 1000, 'Display', 'off', 'StepTolerance', 1e-8, 'OptimalityTolerance', 1e-8);
        U_opt = fmincon(cost_fun, U, [], [], [], [], lower_bounds, upper_bounds, [], options);
        
        % Use only the first control action from the optimal sequence
        u = U_opt(1:n);
        u_body = u;
    else
        % MPC with Desired State Information as State Estimates (if dynamics uncertain)
        H = max(1, round(prediction_horizon / 2)); % Use a reduced prediction horizon
        step_size_pred = step_size / H;
        
        % Define the initial guess for the control sequence over the reduced horizon
        if i > 1
            U = repmat(u_out(:, i-1), H, 1);
        else
            U = repmat(zeros(n, 1), H, 1);
        end
        
        % Expand the bounds to cover the reduced prediction horizon
        lower_bounds = repmat(-u_bound, H, 1);
        upper_bounds = repmat(u_bound, H, 1);
        
        % Use the desired trajectory as state estimates
        xi_pred = xdi;
        xi_dot_pred = xdi_dot;
        
        % Define cost function for the reduced horizon
        cost_fun = @(U) compute_horizon_cost_with_desired_state(U, xi_pred, xi_dot_pred, t, A, B, f1, Q_ei, Q_ri, R, alpha_1, step_size_pred, H);
        
        % Solve optimization problem using fmincon
        U_opt = fmincon(cost_fun, U, [], [], [], [], lower_bounds, upper_bounds, [], options);
        
        % Use only the first control action from the optimal sequence
        u = U_opt(1:n);
        u_body = u;
    end
    
    % System State Update using Euler Integration
    [f, g] = dynamics(Xi); % Removed unused Jacobian J to improve computational efficiency
    Xdot = [xi_dot; f] + [zeros(n, 1); g * u];
    X(:, i + 1) = Xi + step_size * Xdot;
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
        xdi_h = [A*cos(f1*t_h); A*sin(f1*t_h); B*f1*t_h; 0; 0; -0.5*B*f1*t_h];
        xdi_dot_h = [-A*f1*sin(f1*t_h); A*f1*cos(f1*t_h); B*f1; 0; 0; -0.5*B*f1];
        
        % Extract control input for the current time step
        u_h = U((h-1)*length(xi) + 1 : h*length(xi));

        % Update predicted states
        [f, g] = dynamics([xi_pred; xi_dot_pred]);
        xi_dot_pred = xi_dot_pred + step_size_pred * (f + g * u_h);
        xi_pred = xi_pred + step_size_pred * xi_dot_pred;

        % Calculate predicted error
        ei_pred = xi_pred - xdi_h;
        ri_pred = xi_dot_pred + alpha_1 * ei_pred;

        % Accumulate cost
        cost = cost + (ei_pred' * Q_ei * ei_pred) + (u_h' * R * u_h) + (ri_pred' * Q_ri * ri_pred);
    end
end

function cost = compute_horizon_cost_with_desired_state(U, xi_pred, xi_dot_pred, t, A, B, f1, Q_ei, Q_ri, R, alpha_1, step_size_pred, H)
    cost = 0;

    for h = 1:H
        % Time update for reference trajectory
        t_h = t + (h - 1) * step_size_pred;
        
        % Desired trajectory at time t_h
        xdi_h = [A*cos(f1*t_h); A*sin(f1*t_h); B*f1*t_h; 0; 0; -0.5*B*f1*t_h];
        xdi_dot_h = [-A*f1*sin(f1*t_h); A*f1*cos(f1*t_h); B*f1; 0; 0; -0.5*B*f1];
        
        % Extract control input for the current time step
        u_h = U((h-1)*length(xi_pred) + 1 : h*length(xi_pred));

        % Update predicted states using simple open-loop integration
        xi_dot_pred = xi_dot_pred + step_size_pred * u_h; % Using u_h directly as system input
        xi_pred = xi_pred + step_size_pred * xi_dot_pred;

        % Calculate predicted error
        ei_pred = xi_pred - xdi_h;
        ri_pred = xi_dot_pred + alpha_1 * ei_pred;

        % Accumulate cost
        cost = cost + (ei_pred' * Q_ei * ei_pred) + (u_h' * R * u_h) + (ri_pred' * Q_ri * ri_pred);
    end
end

% Note: The 'fmincon' function is used to solve the optimization problem for MPC.
% You will need the Optimization Toolbox in MATLAB for this to work.
% Replace the dynamics() function with the actual function that defines system dynamics.
