clear;
clc;
clear functions;
function dxdt = mySystem(t, x)

    % System parameters
    g = 9.81;
    m = 1;
    I = 0.1;
    r = 0.5;   % example value
    
    % Extract states
    x1 = x(1);  % theta
    x2 = x(2);  % theta_dot
    x3 = x(3);  % alpha
    x4 = x(4);  % alpha_dot
    
    % Desired outputs
    y1_des = pi;
    y1_dot_des = 0;
    y2_des = pi/2;
    y2_dot_des = 0;
    
    % Actual outputs
    y1 = x1;
    y1_dot = x2;
    y2 = x3;
    y2_dot = x4;
    
    % Control gains
    kp1 = 20; kd1 = 5; ki1 = 1;
    kp2 = 10; kd2 = 5; ki2 = 1;
    
    % Persistent variables for integral errors
    persistent int_err1 int_err2 last_t
    
    if isempty(int_err1)
        int_err1 = 0;
        int_err2 = 0;
        last_t = t;
    end
    
    dt = t - last_t;
    last_t = t;
    
    % Update integral errors
    e1 = y1_des - y1;
    e2 = y2_des - y2;
    
    int_err1 = int_err1 + e1 * dt;
    int_err2 = int_err2 + e2 * dt;
    
    % Virtual control law with integral
    v1 = kd1*(y1_dot_des - y1_dot) + kp1*(y1_des - y1) + ki1*int_err1;
    v2 = kd2*(y2_dot_des - y2_dot) + kp2*(y2_des - y2) + ki2*int_err2;
    
    v = [v1; v2];
    
    % Define f and g matrices
    f = [x2;
         -g*sin(x1);
         x4;
         0];
    
    g_mat = [0,           0,           0;
             -sin(x1-x3)/m, -sin(x1-x3)/m, 0;
             0,           0,           0;
             -r,           r,           I];
    
    % Outputs: h = [theta, alpha]
    % Lie derivatives
    Lfh1 = x2;
    Lfh2 = x4;
    
    L2fh1 = -g*sin(x1);
    L2fh2 = 0;
    
    LgLfh1 = [-sin(x1-x3)/m, -sin(x1-x3)/m, 0];
    LgLfh2 = [-r, r, I];
    
    b = [L2fh1;
         L2fh2];
    
    A = [LgLfh1;
         LgLfh2];
    
    % Solve for control input
    u = pinv(A)*(v - b)
    
    % Compute dxdt
    dxdt = f + g_mat*u;
end

% Initial condition
x0 = [0; 0; 0.0; 0];  % small initial angles

% Simulate
[t, x] = ode45(@mySystem, [0 10], x0);

% Plot
figure;
plot(t, x(:,1), 'r', 'DisplayName', 'x1 (theta)');
hold on;
plot(t, x(:,3), 'b', 'DisplayName', 'x3 (alpha)');
xlabel('Time (s)');
ylabel('States');
legend;
title('Feedback Linearization: State trajectories');
grid on;
disp("done")