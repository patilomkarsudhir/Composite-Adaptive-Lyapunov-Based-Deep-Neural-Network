clear all

n=2; %Size of States

k=10; %Number of hidden layers
L=10; %Number of neurons in each layer

L_in=2*n;
L_out=n;

L_vec_in=(L_in+1)*L; %Length of vectorized input layer weights
L_vec_mid=(L+1)*L; %Length of vectorized intermediate layer weights
L_vec_out=(L+1)*L_out; %Length of vectorized output layer weights
L_vec=L_vec_in+(k-1)*L_vec_mid+L_vec_out; %Total vectorized weight length

X_init=[-1;1;0;0];  %Initial State
theta_init=0.5*(-1+2*rand(L_vec,1));  %Initial Theta

% Gains
alpha_1=5;
alpha_2=10;
alpha_3=20;

kr=5;
kf=20;
k_theta=0.0001;

gamma_init=1*eye(L_vec);   %Gamma Initial

beta_0=10;

step_size=0.01;

simtime=100;  %Simulation Time

omega=pi;   %Omega
N=100;
X_rand=0.25*(-1+2*rand(4,1,N)); %Random Points for Generating Test Dataset

% Measurement noise parameters based on Signal-to-Noise Ratio (SNR)
SNR = 50; % Signal-to-Noise Ratio in dB

% Run Simulations
[e_trad,DNN_error_trad,u_trad,theta_trad,F_error_rms_trad]=DNNSim("Traditional",k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,X_rand); 
[e_comp,DNN_error_comp,u_comp,theta_comp,F_error_rms_comp,ftilde_comp,time]=DNNSim("Composite",k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,X_rand); 

[e_o,u_out_o,ftilde_o,time_o,X_o,Xd_o] = ObserverSim(n,step_size,simtime,X_init,kr,kf,alpha_1,alpha_2,SNR);

% PID Control Simulation
[e_pid, u_out_pid, time_pid, X_pid, Xd_pid] = PIDSim(n, step_size, simtime, X_init, kr, alpha_1, SNR);

% MPC Control Simulation
prediction_horizon = 3; % Define the prediction horizon for MPC
Q_ei = eye(n); % State error weight for MPC
Q_ri = eye(n); % Rate error weight for MPC
R = 0.0001* eye(n); % Control effort weight for MPC
u_bound = 100 * ones(n, 1); % Control bounds for MPC


[e_mpc, u_out_mpc, time_mpc, X_mpc, Xd_mpc] = MPCSim(n, step_size, simtime, X_init, alpha_1, prediction_horizon, Q_ei, Q_ri, R, u_bound);

%% Plotting Tracking Error and Control Input Comparison
figure;

% Tracking Error Comparison
subplot(2,1,1);
plot(time, (180/pi)*vecnorm(e_trad), 'LineWidth', 1.5);
hold on;
plot(time, (180/pi)*vecnorm(e_comp), 'LineWidth', 1.5);
plot(time_o, (180/pi)*vecnorm(e_o), 'LineWidth', 1.5);
plot(time_pid, (180/pi)*vecnorm(e_pid), 'LineWidth', 1.5);
plot(time_mpc, (180/pi)*vecnorm(e_mpc), 'LineWidth', 1.5);
hold off;

ylabel('Tracking Error Norm (deg)', 'Interpreter', 'latex');
xlabel('Time (s)');
legend({'Tracking Error-Based', 'Composite', 'Observer-Based', 'PID', 'Nonlinear MPC'}, 'Location', 'northeast');
title('Tracking Error Comparison');
grid on;

% Add zoomed inset for Tracking Error
axes('Position', [0.6 0.7 0.25 0.2]); % Adjust position and size as needed
box on;
plot(time, (180/pi)*vecnorm(e_trad), 'LineWidth', 1.5);
hold on;
plot(time, (180/pi)*vecnorm(e_comp), 'LineWidth', 1.5);
plot(time_o, (180/pi)*vecnorm(e_o), 'LineWidth', 1.5);
plot(time_pid, (180/pi)*vecnorm(e_pid), 'LineWidth', 1.5);
plot(time_mpc, (180/pi)*vecnorm(e_mpc), 'LineWidth', 1.5);
hold off;
xlim([10 20]); % Set zoomed range
grid on;

% Control Input Comparison
subplot(2,1,2);
plot(time, vecnorm(u_trad), 'LineWidth', 1.5);
hold on;
plot(time, vecnorm(u_comp), 'LineWidth', 1.5);
plot(time_o, vecnorm(u_out_o), 'LineWidth', 1.5);
plot(time_pid, vecnorm(u_out_pid), 'LineWidth', 1.5);
plot(time_mpc, vecnorm(u_out_mpc), 'LineWidth', 1.5);
hold off;

ylabel('Control Input Norm (Nm)');
xlabel('Time (s)');
legend({'Tracking Error-Based', 'Composite', 'Observer-Based', 'PID', 'Nonlinear MPC'}, 'Location', 'northeast');
title('Control Input Comparison');
grid on;

% Add zoomed inset for Control Input
axes('Position', [0.6 0.25 0.25 0.2]); % Adjust position and size as needed
box on;
plot(time, vecnorm(u_trad), 'LineWidth', 1.5);
hold on;
plot(time, vecnorm(u_comp), 'LineWidth', 1.5);
plot(time_o, vecnorm(u_out_o), 'LineWidth', 1.5);
plot(time_pid, vecnorm(u_out_pid), 'LineWidth', 1.5);
plot(time_mpc, vecnorm(u_out_mpc), 'LineWidth', 1.5);
hold off;
xlim([10 20]); % Set zoomed range
grid on;


%% Summary Statistics
rms_e_trad = rms(vecnorm(e_trad(:, 5000:10000))) * 180 / pi;
rms_e_comp = rms(vecnorm(e_comp(:, 5000:10000))) * 180 / pi;
rms_e_o = rms(vecnorm(e_o(:, 5000:10000))) * 180 / pi;
rms_e_pid = rms(vecnorm(e_pid(:, 5000:10000))) * 180 / pi;
rms_e_mpc = rms(vecnorm(e_mpc(:, 5000:10000))) * 180 / pi;

rms_u_trad = rms(vecnorm(u_trad));
rms_u_comp = rms(vecnorm(u_comp));
rms_u_o = rms(vecnorm(u_out_o));
rms_u_pid = rms(vecnorm(u_out_pid));
rms_u_mpc = rms(vecnorm(u_out_mpc));

% Create comparison table
ComparisonTable = table([rms_e_trad; rms_e_comp; rms_e_o; rms_e_pid; rms_e_mpc], ...
                        [rms_u_trad; rms_u_comp; rms_u_o; rms_u_pid; rms_u_mpc], ...
                        'RowNames', {'Tracking_Error_Based', 'Composite', 'Observer_Based', 'PID', 'MPC'}, ...
                        'VariableNames', {'RMS_Tracking_Error','RMS_Control_Input'});

% Display the comparison table
disp(ComparisonTable);

%% Improved Plots
% Figure for RMS Tracking Errors
figure;
bar(categorical(ComparisonTable.Properties.RowNames), ComparisonTable.RMS_Tracking_Error, 'FaceColor', 'flat');

ylabel('RMS Tracking Error (deg)');
title('RMS Tracking Error for Different Controllers');
grid on;

% Figure for RMS Control Inputs
figure;
bar(categorical(ComparisonTable.Properties.RowNames), ComparisonTable.RMS_Control_Input, 'FaceColor', 'flat');

ylabel('RMS Control Input');
title('RMS Control Input for Different Controllers');
grid on;
