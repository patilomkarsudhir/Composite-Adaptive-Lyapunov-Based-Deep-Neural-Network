% Assuming `time`, `X_trad`, `X_comp`, `X_o`, `X_pid`, `X_mpc`, `xd`, and `intervals` are loaded or calculated before

% Filter data for time range 0 to 9 seconds
time_range = (time >= 0 & time <= 9);

% Extract data within the time range
time_filtered = time(time_range);
X_trad_filtered = X_trad(:, time_range);
X_comp_filtered = X_comp(:, time_range);
X_o_filtered = X_o(:, time_range);
X_pid_filtered = X_pid(:, time_range);
X_mpc_filtered = X_mpc(:, time_range);
xd_filtered = xd(:, time_range);

% Filter intervals for feedback lost/regained markers
intervals_filtered = intervals(intervals(:, 1) <= 7, :);

% Plot the 3D trajectories
figure(4);
hold on;

% Plot filtered trajectories
plot3(X_trad_filtered(1, :), X_trad_filtered(2, :), X_trad_filtered(3, :), 'DisplayName', 'Tracking Error-Based', 'LineWidth', 2);
plot3(X_comp_filtered(1, :), X_comp_filtered(2, :), X_comp_filtered(3, :), 'DisplayName', 'Composite', 'LineWidth', 1.5);
plot3(X_o_filtered(1, :), X_o_filtered(2, :), X_o_filtered(3, :), 'g-.', 'DisplayName', 'Observer-Based', 'LineWidth', 1.5);
plot3(X_pid_filtered(1, :), X_pid_filtered(2, :), X_pid_filtered(3, :), 'm--', 'DisplayName', 'Nonlinear PID', 'LineWidth', 1.5);
plot3(X_mpc_filtered(1, :), X_mpc_filtered(2, :), X_mpc_filtered(3, :), 'k:', 'DisplayName', 'Nonlinear MPC', 'LineWidth', 1.5);

% Plot filtered reference trajectory
plot3(xd_filtered(1, :), xd_filtered(2, :), xd_filtered(3, :), 'k--', 'DisplayName', 'Reference Trajectory', 'LineWidth', 2);

% Add markers for feedback lost and regained
feedbackLostLegendAdded = false;
feedbackRegainedLegendAdded = false;

for i = 1:size(intervals_filtered, 1)
    % Indices for feedback intervals within the filtered time
    start_idx = find(time >= intervals_filtered(i, 1), 1, 'first');
    end_idx = find(time >= intervals_filtered(i, 2), 1, 'first');
    
    % Mark feedback lost and regained for each trajectory
    trajectories = {X_trad, X_comp, X_o, X_pid, X_mpc};
    for j = 1:length(trajectories)
        X = trajectories{j};
        
        % Feedback lost marker
        if ~feedbackLostLegendAdded && j == 1 % Add legend only for the first trajectory
            plot3(X(1, start_idx), X(2, start_idx), X(3, start_idx), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Feedback Lost');
            feedbackLostLegendAdded = true;
        else
            plot3(X(1, start_idx), X(2, start_idx), X(3, start_idx), 'rx', 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
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

hold off;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Z (m)');
legend('show');
title('Trajectory Visualization with Feedback Loss/Regain Markers (0 to 9 seconds)');
grid on;
