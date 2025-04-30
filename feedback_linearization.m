% Feedback Linearization: Relative Degree Calculation
clear all; close all; clc;

% System parameters
g = 9.81;      % Gravity (m/s^2)
l = 1;         % Length (m)
m = 1;         % Mass (kg)
I = 0.1;       % Moment of inertia (kg*m^2)

% Define symbolic variables
syms x1 x2 x3 x4 r real     % State variables
syms F_l F_r tau_a real     % Inputs
syms v1 v2 real             % New (virtual) inputs

% State and input vectors
x = [x1; x2; x3; x4];
u = [F_l; F_r; tau_a];

% System dynamics
f = [x2;
     -(g*sin(x1));
     x4;
     0];

g_mat = [0,           0,           0;
         -sin(x1-x3)/(m), -sin(x1-x3)/(m), 0;
          0,           0,           0;
         -r,           r,           I];

% ---- Define outputs ----
% Example: let's try to linearize theta (x1) and alpha (x3)
h = [x1; x3];   % Outputs


% ---- Compute Relative Degree ----
Lfh1 = jacobian(h(1), x) * f
Lgh1 = jacobian(h(1), x) * g_mat
L2fh1 = jacobian(Lfh1, x) * f
LgLfh1 = jacobian(Lfh1, x) * g_mat
%r = 2

Lfh2 = jacobian(h(2), x) * f
Lgh2 = jacobian(h(2), x) * g_mat
L2fh2 = jacobian(Lfh2, x) * f
LgLfh2 = jacobian(Lfh2, x) * g_mat
%r = 2

% Lfh3 = jacobian(h(3), x) * f
% Lgh3 = jacobian(h(3), x) * g_mat
% L2fh3 = jacobian(Lfh3, x) * f
% LgLfh3 = jacobian(Lfh3, x) * g_mat
% %r = 2
% 
% Lfh4 = jacobian(h(4), x) * f
% Lgh4 = jacobian(h(4), x) * g_mat
% L2fh4 = jacobian(Lfh4, x) * f
% LgLfh4 = jacobian(Lfh4, x) * g_mat
% %r = 1


% % ---- Display results ----
disp('Total relative degree (sum): 6');


% inversion
% g_inv = pinv(g_mat)

% Now assemble
b = [L2fh1;
     L2fh2];    % b(x)

A = [LgLfh1;
     LgLfh2];   % A(x)

disp('b(x) =')
pretty(b)

disp('A(x) =')
pretty(A)

v_des = [3.14; 3.14/2]
A_inv = pinv(A)
u_out = A_inv * (v_des-b)
