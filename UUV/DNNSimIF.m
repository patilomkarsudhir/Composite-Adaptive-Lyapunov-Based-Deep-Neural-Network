function [e,DNN_error,u_out,theta_out,ftilde,time,X,Xd] = DNNSimIF(Method,k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,intervals,SNR)
% This function simulates a system using a deep neural network (DNN) for control and state estimation with noisy state measurements.

L_in = 2 * n;
L_out = n;
L_vec_in = (L_in + 1) * L; % Length of vectorized input layer weights
L_vec_mid = (L + 1) * L; % Length of vectorized intermediate layer weights
L_vec_out = (L + 1) * L_out; % Length of vectorized output layer weights
L_vec = L_vec_in + (k - 1) * L_vec_mid + L_vec_out; % Total vectorized weight length

time_length = simtime / step_size;

X = X_init;
theta = theta_init;
Gamma = gamma_init;

kappa_0 = 2 * max(abs(eig(gamma_init)));

time = zeros(1, time_length);
Xd = zeros(2 * n, time_length);
e = zeros(n, time_length);
ftilde = zeros(n, time_length);
DNN_error = zeros(n, time_length);
u_out = zeros(n, time_length);
theta_out = zeros(L_vec, time_length);
Xhat = zeros(2 * n, time_length);

act = 'tanh';



for i = 1:time_length
    A = 2;
    B = 0.5;
    f1 = 0.5;
    
    t = (i - 1) * step_size;       % Time   
    time(i) = t;
    
    % Vectorized condition to determine if t is outside all intervals
    flag = all(t <= intervals(:, 1) | t >= intervals(:, 2));
    
    Xi = X(:, i);  
    xi = X(1:n, i);               % State (positions)
    xi_dot = X(n+1:2*n, i);       % State (velocities)
    
    % Desired Trajectory
    xdi = [A*cos(f1*t); A*sin(f1*t); B*f1*t; 0; 0; -0.5*B*f1*t];
    xdi_dot = [-A*f1*sin(f1*t); A*f1*cos(f1*t); B*f1; 0; 0; -0.5*B*f1];
    xdi_ddot = [-A*f1^2*cos(f1*t); -A*f1^2*sin(f1*t); 0; 0; 0; 0];
    
    % Calculate noise standard deviation based on SNR
    signal_power = mean(xi.^2) + 1e-6; % Adding a small positive constant to avoid numerical issues when signal power is very low
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
    
    if flag
        [Phi, Phi_prime] = DNNGrad(k, L, L_in, L_out, noisy_Xi, theta, act);

        if i == 1
            rhat_i = ri;
            fhat_i = zeros(n, 1);
            fhat_int = zeros(n, 1);
        end    

        E_i = fhat_i - Phi;

        if Method == "Traditional"
            thetadot = -k_theta * Gamma * theta + Gamma * Phi_prime' * noisy_ri;
        else
            beta = beta_0 * (1 - norm(Gamma) / kappa_0);
            Phi_prime_gram = Phi_prime' * Phi_prime;
            Gammadot = beta * Gamma - Gamma * Phi_prime_gram * Gamma;
            Gamma = Gamma + step_size * Gammadot;
            % Clamp or project Gamma to ensure numerical stability
            Gamma = max(Gamma, 1e-6); % Ensure Gamma does not drop below a small positive value
            thetadot = -k_theta * Gamma * theta + Gamma * Phi_prime' * (noisy_ri + alpha_3 * E_i);
        end

        theta = theta + step_size * thetadot;   

        [~, g] = dynamics(noisy_Xi);
        u = (g' * g + 1e-6 * eye(size(g, 2))) \ (g' * (xdi_ddot - Phi - (alpha_1 + kr) * noisy_ri + (alpha_1^2 - 1) * noisy_ei));       

        rhatdot = g * u - xdi_ddot + fhat_i + alpha_2 * (noisy_ri - rhat_i) + alpha_1 * (noisy_ri - alpha_1 * noisy_ei);
        fhat_int_dot = (kf * alpha_2 + 1) * (noisy_ri - rhat_i);

        rhat_i = rhat_i + step_size * rhatdot;
        fhat_int = fhat_int + step_size * fhat_int_dot;

        fhat_i = kf * (noisy_ri - rhat_i) + fhat_int;
        Xhati = Xi;
        Xhat(:, i) = Xhati;
    else
        [Phi] = DNNGrad(k, L, L_in, L_out, Xhati, theta, act);
        
        [~, gh] = dynamics(Xhati);
        u = (gh' * gh + 1e-6 * eye(size(gh, 2))) \ (gh' * (xdi_ddot - Phi));
        xhatdot = Xhati(n+1:2*n);
        xhatdotdot = Phi + gh * u;
        Xhatdot = [xhatdot; xhatdotdot];
        Xhati = Xhati + step_size * Xhatdot;
        Xhat(:, i) = Xhati;       
    end
    
    [f, g, J] = dynamics(Xi);
    u_body = J' * u;
    Xdot = [xi_dot; f] + [zeros(n, 1); g * u];
    X(:, i + 1) = Xi + step_size * Xdot;

    ftilde(:, i) = f - fhat_i;
    DNN_error(:, i) = f - Phi;
    u_out(:, i) = u_body;
    theta_out(:, i) = theta;
    e(:, i) = ei;
end
