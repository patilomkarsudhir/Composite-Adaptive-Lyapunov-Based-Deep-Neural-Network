close all
clear all

n=6; %Size of States

k=5; %Number of hidden layers
L=5; %Number of neurons in each layer

L_in=2*n;
L_out=n;

L_vec_in=(L_in+1)*L; %Length of vectorized input layer weights
L_vec_mid=(L+1)*L; %Length of vectorized intermediate layer weights
L_vec_out=(L+1)*L_out; %Length of vectorized output layer weights
L_vec=L_vec_in+(k-1)*L_vec_mid+L_vec_out; %Total vectorized weight length



X_init=[-0.5;-0.5;-0.5;0;0;0;zeros(6,1)];  %Initial State
theta_init=0.5*(-1+2*rand(L_vec,1));  %Initial Theta

%Gains

alpha_1=10;
alpha_2=20;
alpha_3=80;

kr=20;
kf=20;
k_theta=0.0001;


gamma_init=0.25*eye(L_vec);   %Gamma Initial

beta_0=1;

step_size=0.01;

simtime=30;  %Simulation Time

N=100;

intervals = [5 6;7 8;9 10;11 12;14 15;17 18;19 20;21 22;24 25;26 27]; % Each row is an interval [start, end]

% Measurement noise parameters based on Signal-to-Noise Ratio (SNR)
SNR = 50; % Signal-to-Noise Ratio in dB

[e_trad,DNN_error_trad,u_trad,theta_trad,ftilde_trad,time,X_trad]=DNNSimIF("Traditional",k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,intervals,SNR);
[e_comp,DNN_error_comp,u_comp,theta_comp,ftilde_comp,time,X_comp,xd]=DNNSimIF("Composite",k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,intervals,SNR);
[e_o,u_out_o,ftilde_o,time_o,X_o,Xd_o] = ObserverSimIF(n,step_size,simtime,X_init,kr,kf,alpha_1,alpha_2,intervals,SNR);

% PID Control Simulation
[e_pid, u_out_pid, time_pid, X_pid, Xd_pid] = PIDSimIF(n, step_size, simtime, X_init, kr, alpha_1, intervals,SNR);

% MPC Control Simulation
prediction_horizon = 3; % Define the prediction horizon for MPC
Q_ei = eye(n); % State error weight for MPC
Q_ri = eye(n); % Rate error weight for MPC
R = 10*eye(n); % Control effort weight for MPC
u_bound = 5 * ones(n, 1); % Control bounds for MPC

[e_mpc, u_out_mpc, time_mpc, X_mpc, Xd_mpc] = MPCSimIF(n, step_size, simtime, X_init, kr, alpha_1, intervals, prediction_horizon, Q_ei, Q_ri, R, u_bound);



%%

figure(1)

% Define intervals

% First subplot: Linear Tracking Error
subplot(2,1,1)
plot(time, vecnorm(e_trad(1:3,:)), 'DisplayName', 'Tracking Error-Based','LineWidth', 2);
hold on;
plot(time, vecnorm(e_comp(1:3,:)), 'DisplayName', 'Composite','LineWidth', 1.5);
plot(time, vecnorm(e_o(1:3,:)), 'g-.', 'DisplayName', 'Observer-Based','LineWidth', 1.5);
plot(time_pid, vecnorm(e_pid(1:3,:)), 'm--', 'DisplayName', 'Nonlinear PD','LineWidth', 1.5);
plot(time_mpc, vecnorm(e_mpc(1:3,:)), 'k:', 'DisplayName', 'Nonlinear MPC','LineWidth', 1.5);

% Add patches for intervals
for i = 1:size(intervals, 1)
    x_patch = [intervals(i,1), intervals(i,2), intervals(i,2), intervals(i,1)];
    y_patch = [min(ylim()), min(ylim()), max(ylim()), max(ylim())];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
hold off;

ylabel('Linear Tracking Error', 'Interpreter', 'latex')
xlabel('Time (s)')
legend;

% Second subplot: Angular Tracking Error
subplot(2,1,2)
plot(time, vecnorm(e_trad(4:6,:)), 'DisplayName', 'Tracking Error-Based','LineWidth', 2);
hold on;
plot(time, vecnorm(e_comp(4:6,:)), 'DisplayName', 'Composite','LineWidth', 1.5);
plot(time, vecnorm(e_o(4:6,:)), 'g-.', 'DisplayName', 'Observer-Based','LineWidth', 1.5);
plot(time_pid, vecnorm(e_pid(4:6,:)), 'm--', 'DisplayName', 'Nonlinear PD','LineWidth', 1.5);
plot(time_mpc, vecnorm(e_mpc(4:6,:)), 'k:', 'DisplayName', 'Nonlinear MPC','LineWidth', 1.5);

% Add patches
for i = 1:size(intervals, 1)
    x_patch = [intervals(i,1), intervals(i,2), intervals(i,2), intervals(i,1)];
    y_patch = [min(ylim()), min(ylim()), max(ylim()), max(ylim())];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8 0.8 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
hold off;

ylabel('Angular Tracking Error', 'Interpreter', 'latex')
xlabel('Time (s)')
legend;

% 

%% 
figure(2);

% First Subplot
subplot(2, 1, 1);
plot(time, vecnorm(u_trad(1:3, :)), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);
hold on;
plot(time, vecnorm(u_comp(1:3, :)), 'DisplayName', 'Composite', 'LineWidth', 1.5);
plot(time, vecnorm(u_out_o(1:3, :)), 'g-.', 'DisplayName', 'Observer-Based', 'LineWidth', 1.5);
plot(time_pid, vecnorm(u_out_pid(1:3, :)), 'm--', 'DisplayName', 'Nonlinear PD', 'LineWidth', 1.5);
plot(time_mpc, vecnorm(u_out_mpc(1:3, :)), 'k:', 'DisplayName', 'Nonlinear MPC', 'LineWidth', 1.5);
ylabel('Linear Control Norm (N)');
xlabel('Time (s)');

% Manually set the limits for the y-axis to prevent scaling issues with patches
ylim1 = [0, max([vecnorm(u_trad(1:3, :)), vecnorm(u_comp(1:3, :)), vecnorm(u_out_o(1:3, :)), vecnorm(u_out_pid(1:3, :)), vecnorm(u_out_mpc(1:3, :))]) + 0.5];
ylim(ylim1);

% Add patches for intervals to the first subplot
for i = 1:size(intervals, 1)
    x_patch = [intervals(i, 1), intervals(i, 2), intervals(i, 2), intervals(i, 1)];
    y_patch = [ylim1(1), ylim1(1), ylim1(2), ylim1(2)];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
legend;
hold off;

% Second Subplot
subplot(2, 1, 2);
plot(time, vecnorm(u_trad(4:6, :)), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);
hold on;
plot(time, vecnorm(u_comp(4:6, :)), 'DisplayName', 'Composite', 'LineWidth', 1.5);
plot(time, vecnorm(u_out_o(4:6, :)), 'g-.', 'DisplayName', 'Observer-Based', 'LineWidth', 1.5);
plot(time_pid, vecnorm(u_out_pid(4:6, :)), 'm--', 'DisplayName', 'Nonlinear PD', 'LineWidth', 1.5);
plot(time_mpc, vecnorm(u_out_mpc(4:6, :)), 'k:', 'DisplayName', 'Nonlinear MPC', 'LineWidth', 1.5);
ylabel('Angular Control Norm (Nm)');
xlabel('Time (s)');

% Manually set the limits for the y-axis in the second subplot
ylim2 = [0, max([vecnorm(u_trad(4:6, :)), vecnorm(u_comp(4:6, :)), vecnorm(u_out_o(4:6, :)), vecnorm(u_out_pid(4:6, :)), vecnorm(u_out_mpc(4:6, :))]) + 0.1];
ylim(ylim2);

% Add patches for intervals to the second subplot
for i = 1:size(intervals, 1)
    x_patch = [intervals(i, 1), intervals(i, 2), intervals(i, 2), intervals(i, 1)];
    y_patch = [ylim2(1), ylim2(1), ylim2(2), ylim2(2)];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
legend;
hold off;

%% 



figure(3);

% First Subplot
subplot1 = subplot(2, 1, 1);
plot(time, vecnorm(DNN_error_trad(1:3, :)), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);
hold on;
plot(time, vecnorm(DNN_error_comp(1:3, :)), 'DisplayName', 'Composite', 'LineWidth', 1.5);
plot(time, vecnorm(ftilde_o(1:3, :)), 'g-.', 'DisplayName', 'Observer-Based Disturbance Rejection', 'LineWidth', 1.5);
ylabel('Linear Dynamics Error (m/s^2)');
xlabel('Time (s)');

% Manually set the limits for the y-axis to prevent scaling issues with patches
ylim1 = [0, max([vecnorm(DNN_error_trad(1:3, :)), vecnorm(DNN_error_comp(1:3, :)), vecnorm(ftilde_o(1:3, :))])+50];
ylim(ylim1);

% Add patches for intervals to the first subplot
for i = 1:size(intervals, 1)
    x_patch = [intervals(i, 1), intervals(i, 2), intervals(i, 2), intervals(i, 1)];
    y_patch = [ylim1(1), ylim1(1), ylim1(2), ylim1(2)];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
legend;
hold off;

% Inset for the first 0.2 seconds in the first subplot
insetPosition1 = get(subplot1, 'Position');
insetPosition1 = [insetPosition1(1) + 0.2 * insetPosition1(3), ... % Adjust x position
                  insetPosition1(2) + 0.3 * insetPosition1(4), ... % Adjust y position
                  0.2 * insetPosition1(3), 0.3 * insetPosition1(4)]; % Adjust size
inset1 = axes('Position', insetPosition1); % Position relative to the whole figure
hold(inset1, 'on');
plot(inset1, time, vecnorm(DNN_error_trad(1:3, :)), 'LineWidth', 2);
plot(inset1, time, vecnorm(DNN_error_comp(1:3, :)), 'LineWidth', 1.5);
plot(inset1, time, vecnorm(ftilde_o(1:3, :)), 'g-.', 'LineWidth', 1.5);
xlim(inset1, [0, 0.2]);
ylim(inset1, [0, max(vecnorm(ftilde_o(1:3, 1:find(time <= 0.2, 1, 'last'))))]);
box(inset1, 'on'); % Add border
hold(inset1, 'off');

% Second Subplot
subplot2 = subplot(2, 1, 2);
plot(time, vecnorm(DNN_error_trad(4:6, :)), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);
hold on;
plot(time, vecnorm(DNN_error_comp(4:6, :)), 'DisplayName', 'Composite', 'LineWidth', 1.5);
plot(time, vecnorm(ftilde_o(4:6, :)), 'g-.', 'DisplayName', 'Observer-Based Disturbance Rejection', 'LineWidth', 1.5);
ylabel('Angular Dynamics Error (rad/s^2)');
xlabel('Time (s)');

% Manually set the limits for the y-axis in the second subplot
ylim2 = [0, max([vecnorm(DNN_error_trad(4:6, :)), vecnorm(DNN_error_comp(4:6, :)), vecnorm(ftilde_o(4:6, :))])+5];
ylim(ylim2);

% Add patches for intervals to the second subplot
for i = 1:size(intervals, 1)
    x_patch = [intervals(i, 1), intervals(i, 2), intervals(i, 2), intervals(i, 1)];
    y_patch = [ylim2(1), ylim2(1), ylim2(2), ylim2(2)];
    if i == 1 % Only add DisplayName to one patch
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'DisplayName', 'Feedback Denied Zone');
    else
        patch(x_patch, y_patch, [0.8, 0.8, 0.8], 'FaceAlpha', 0.5, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
legend;
hold off;

% Inset for the first 0.2 seconds in the second subplot
insetPosition2 = get(subplot2, 'Position');
insetPosition2 = [insetPosition2(1) + 0.2 * insetPosition2(3), ... % Adjust x position
                  insetPosition2(2) + 0.3 * insetPosition2(4), ... % Adjust y position
                  0.2 * insetPosition2(3), 0.3 * insetPosition2(4)]; % Adjust size
inset2 = axes('Position', insetPosition2); % Position relative to the whole figure
hold(inset2, 'on');
plot(inset2, time, vecnorm(DNN_error_trad(4:6, :)), 'LineWidth', 2);
plot(inset2, time, vecnorm(DNN_error_comp(4:6, :)), 'LineWidth', 1.5);
plot(inset2, time, vecnorm(ftilde_o(4:6, :)), 'g-.', 'LineWidth', 1.5);
xlim(inset2, [0, 0.2]);
ylim(inset2, [0, max(vecnorm(ftilde_o(4:6, 1:find(time <= 0.2, 1, 'last'))))]);
box(inset2, 'on'); % Add border
hold(inset2, 'off');

%%

figure(4)
hold on

% Define the intervals where feedback was lost and regained

% Assume 'time' is your time vector and it corresponds to your X_* data

% Tracking Error-Based
plot3(X_trad(1,:), X_trad(2,:), X_trad(3,:), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);

% Composite
plot3(X_comp(1,:), X_comp(2,:), X_comp(3,:), 'DisplayName', 'Composite', 'LineWidth', 1.5);

% Observer-Based
plot3(X_o(1,:), X_o(2,:), X_o(3,:), 'g-.', 'DisplayName', 'Observer-Based Disturbance Rejection', 'LineWidth', 1.5);

% PID-Based
plot3(X_pid(1,:), X_pid(2,:), X_pid(3,:), 'm--', 'DisplayName', 'PID-Based', 'LineWidth', 1.5);

% MPC-Based
plot3(X_mpc(1,:), X_mpc(2,:), X_mpc(3,:), 'k:', 'DisplayName', 'MPC-Based', 'LineWidth', 1.5);

% Reference Trajectory
plot3(xd(1,:), xd(2,:), xd(3,:), 'k--', 'DisplayName', 'Reference Trajectory', 'LineWidth', 2);

% Initialize flags to control legend entry
feedbackLostLegendAdded = false;
feedbackRegainedLegendAdded = false;

% Add markers for feedback lost and regained for each trajectory
for i = 1:size(intervals, 1)
    % Indices for feedback intervals
    start_idx = find(time >= intervals(i, 1), 1, 'first');
    end_idx = find(time >= intervals(i, 2), 1, 'first');

    % Trajectories to mark
    trajectories = {X_trad, X_comp, X_o, X_pid, X_mpc};
    markers = {'rx', 'rx', 'rx', 'rx', 'rx'}; % Keeping the color and marker consistent across trajectories

    for j = 1:length(trajectories)
        X = trajectories{j};
        % Feedback lost marker
        if ~feedbackLostLegendAdded && j == 1 % Add legend only for the first trajectory
            plot3(X(1, start_idx), X(2, start_idx), X(3, start_idx), markers{j}, 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Feedback Lost');
            feedbackLostLegendAdded = true;
        else
            plot3(X(1, start_idx), X(2, start_idx), X(3, start_idx), markers{j}, 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end

        % Feedback regained marker
        if ~feedbackRegainedLegendAdded && j == 1 % Add legend only for the first trajectory
            plot3(X(1, end_idx), X(2, end_idx), X(3, end_idx), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Feedback Regained');
            feedbackRegainedLegendAdded = true;
        else
            plot3(X(1, end_idx), X(2, end_idx), X(3, end_idx), 'bo', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
        end
    end
end

hold off
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
legend('show')


%%
% Calculate RMS values for linear and angular errors
rms_e_trad_lin = rms(vecnorm(e_trad(1:3,:)));
rms_e_comp_lin = rms(vecnorm(e_comp(1:3,:)));
rms_e_o_lin = rms(vecnorm(e_o(1:3,:)));
rms_e_pid_lin = rms(vecnorm(e_pid(1:3,:)));
rms_e_mpc_lin = rms(vecnorm(e_mpc(1:3,:)));

rms_e_trad_ang = rms(vecnorm(e_trad(4:6,:)));
rms_e_comp_ang = rms(vecnorm(e_comp(4:6,:)));
rms_e_o_ang = rms(vecnorm(e_o(4:6,:)));
rms_e_pid_ang = rms(vecnorm(e_pid(4:6,:)));
rms_e_mpc_ang = rms(vecnorm(e_mpc(4:6,:)));

% Calculate RMS values for linear and angular control inputs
rms_u_trad_lin = rms(vecnorm(u_trad(1:3,:)));
rms_u_comp_lin = rms(vecnorm(u_comp(1:3,:)));
rms_u_o_lin = rms(vecnorm(u_out_o(1:3,:)));
rms_u_pid_lin = rms(vecnorm(u_out_pid(1:3,:)));
rms_u_mpc_lin = rms(vecnorm(u_out_mpc(1:3,:)));

rms_u_trad_ang = rms(vecnorm(u_trad(4:6,:)));
rms_u_comp_ang = rms(vecnorm(u_comp(4:6,:)));
rms_u_o_ang = rms(vecnorm(u_out_o(4:6,:)));
rms_u_pid_ang = rms(vecnorm(u_out_pid(4:6,:)));
rms_u_mpc_ang = rms(vecnorm(u_out_mpc(4:6,:)));

% Calculate RMS values for function errors
rms_fe_trad_lin = rms(vecnorm(DNN_error_trad(1:3,:)));
rms_fe_comp_lin = rms(vecnorm(DNN_error_comp(1:3,:)));
rms_fe_o_lin = rms(vecnorm(ftilde_o(1:3,:)));
rms_fe_pid_lin = "N/A";
rms_fe_mpc_lin = "N/A";

rms_fe_trad_ang = rms(vecnorm(DNN_error_trad(4:6,:)));
rms_fe_comp_ang = rms(vecnorm(DNN_error_comp(4:6,:)));
rms_fe_o_ang = rms(vecnorm(ftilde_o(4:6,:)));
rms_fe_pid_ang = "N/A";
rms_fe_mpc_ang = "N/A";

% Create a data matrix with the appropriate orientation
dataMatrix = [rms_e_trad_lin, rms_e_comp_lin, rms_e_o_lin, rms_e_pid_lin, rms_e_mpc_lin; 
              rms_e_trad_ang, rms_e_comp_ang, rms_e_o_ang, rms_e_pid_ang, rms_e_mpc_ang;
              rms_u_trad_lin, rms_u_comp_lin, rms_u_o_lin, rms_u_pid_lin, rms_u_mpc_lin;
              rms_u_trad_ang, rms_u_comp_ang, rms_u_o_ang, rms_u_pid_ang, rms_u_mpc_ang;
              rms_fe_trad_lin, rms_fe_comp_lin, rms_fe_o_lin, rms_fe_pid_lin, rms_fe_mpc_lin;
              rms_fe_trad_ang, rms_fe_comp_ang, rms_fe_o_ang, rms_fe_pid_ang, rms_fe_mpc_ang];

% Create a table from the data matrix
ComparisonTable = array2table(dataMatrix, ...
    'RowNames', {'RMS_Linear_Error', 'RMS_Angular_Error', 'RMS_Linear_Control_Input', 'RMS_Angular_Control_Input', 'RMS_Function_Error_Linear', 'RMS_Function_Error_Angular'}, ...
    'VariableNames', {'Tracking_Error_Based', 'Composite', 'Observer_Based', 'PID_Based', 'MPC_Based'});

disp(ComparisonTable);