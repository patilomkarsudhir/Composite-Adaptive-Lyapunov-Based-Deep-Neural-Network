function [e,u_out,ftilde,time,X,Xd] = ObserverSim(n,step_size,simtime,X_init,kr,kf,alpha_1,alpha_2,SNR)
% This function simulates a system using an observer-based controller for trajectory tracking with noisy state measurements.

% Initialize parameters and states
time_length = simtime / step_size;
X = X_init;
fhat_i = zeros(n, 1);
fhat_int = zeros(n, 1);
rhat_i = zeros(n, 1);


for i = 1:time_length
    A = 2;
    B = 0.5;
    f1 = 0.5;
    
    t = (i - 1) * step_size;       % Time   
    time(i) = t;
    
   
    Xi = X(:, i);  
    xi = X(1:n, i);               % State (positions)
    xi_dot = X(n+1:2*n, i);       % State (velocities)
    
    % Desired Trajectory
    xdi=0.5*0.5*exp(-sin(t))*[sin(t);cos(t)];  %Desired Trajectory
    xdi_dot=0.5*(0.5*exp(-sin(t))*[cos(t);-sin(t)]-0.5*exp(-sin(t))*cos(t)*[sin(t);cos(t)]);
    xdi_ddot=0.5*(-0.5*exp(-sin(t))*cos(t)*[cos(t);-sin(t)]-0.5*exp(-sin(t))*[sin(t);cos(t)]+0.5*exp(-sin(t))*(cos(t)^2)*[sin(t);cos(t)]+0.5*exp(-sin(t))*sin(t)*[sin(t);cos(t)]-0.5*exp(-sin(t))*cos(t)*[cos(t);-sin(t)]);
    
    % Calculate noise standard deviation based on SNR
    signal_power = mean(xi.^2)+ 1e-6;
    noise_power = signal_power / (10^(SNR / 10));
    noise_std_dev = sqrt(noise_power* step_size);
    
    % Add measurement noise to the state
    noisy_xi = xi + noise_std_dev * randn(size(xi));
    noisy_xi_dot = xi_dot + noise_std_dev * randn(size(xi_dot));
    noisy_Xi = [noisy_xi; noisy_xi_dot];

    % Noisy Errors for Control Development
    noisy_ei = noisy_xi - xdi;
    noisy_ei_dot = noisy_xi_dot - xdi_dot;
    noisy_ri = noisy_ei_dot + alpha_1 * noisy_ei;

    % Desired trajectory and errors
    Xdi = [xdi; xdi_dot];
    Xd(:, i) = Xdi;
    ei = xi - xdi;
    e(:, i) = ei;
    
    ei_dot = xi_dot - xdi_dot;
    ri = ei_dot + alpha_1 * ei;
    
    % Observer-based Control

    [~, g] = dynamics(noisy_Xi);
    u = pinv(g) * (xdi_ddot - fhat_i - (alpha_1 + kr) * noisy_ri + (alpha_1^2 - 1) * noisy_ei);

    rhatdot = g * u - xdi_ddot + fhat_i + alpha_2 * (noisy_ri - rhat_i) + alpha_1 * (ri - alpha_1 * noisy_ei);
    fhat_int_dot = (kf * alpha_2 + 1) * (noisy_ri - rhat_i);

    rhat_i = rhat_i + step_size * rhatdot;
    fhat_int = fhat_int + step_size * fhat_int_dot;

    fhat_i = kf * (noisy_ri - rhat_i) + fhat_int;
    Xhati = noisy_Xi;
    Xhat(:, i) = Xhati;

    
    [f, g] = dynamics(Xi);
    Xdot = [xi_dot; f] + [zeros(n, 1); g * u];
    X(:, i + 1) = Xi + step_size * Xdot;

    ftilde(:, i) = f - fhat_i;
    
    u_out(:, i) = u;
    e(:, i) = ei;
end
