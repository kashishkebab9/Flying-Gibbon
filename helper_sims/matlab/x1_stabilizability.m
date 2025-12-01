clear;clc;
% Define symbolic variables
syms x1 x2 x3 x4 F_l F_r tau_a r m I g real

% State and input vectors
x = [x1; x2; x3; x4];
u = [F_l; F_r; tau_a];

% Dynamics
f = [x2;
     -(g*sin(x1)) - (1/m)*(sin(x1 - x3))*(F_l + F_r);
     x4;
     -r*F_l + r*F_r + I*tau_a];

% Jacobians
A_sym = jacobian(f, x);  % ∂f/∂x
B_sym = jacobian(f, u);  % ∂f/∂u

% Substitute constants
m_val = 1.0;
r_val = 0.2;
I_val = 1.0;
g_val = 9.81;

% Range of x1 values to test
x1_vals = linspace(-pi, pi, 200);
x2_val = 0;
x4_val = 0;
x3_val = 0; % Fix x3 to 0 (no relative angle initially)
is_stabilizable = false(size(x1_vals));

for i = 1:length(x1_vals)
    x1_i = x1_vals(i);
    % Define all variables involved in A and B
    all_syms = {x1, x2, x3, x4, F_l, F_r, tau_a, m, r, I, g};
    all_vals = {x1_i, x2_val, x3_val, x4_val, 0, 0, 0, m_val, r_val, I_val, g_val};
    
    A_eval = double(subs(A_sym, all_syms, all_vals));
    B_eval = double(subs(B_sym, all_syms, all_vals));

    % Check eigenvalues of A
    eig_A = eig(A_eval);
    stable_eigs = real(eig_A) <= 0;

    % Controllability matrix
    C = ctrb(A_eval, B_eval);
    rank_C = rank(C);
    
    % Stabilizability test:
    % For each unstable eigenvalue lambda, check (lambda*I - A, B) is controllable
    stabilizable = true;
    for j = 1:length(eig_A)
        if real(eig_A(j)) > 0
            lambda = eig_A(j);
            M = [lambda*eye(size(A_eval)) - A_eval, B_eval];
            if rank(M) < size(A_eval,1)
                stabilizable = false;
                break;
            end
        end
    end

    is_stabilizable(i) = stabilizable;
end

% Plotting results
figure;
plot(x1_vals, is_stabilizable, 'LineWidth', 2);
xlabel('$x_1$ (rad)', 'Interpreter', 'latex');
ylabel('Stabilizable', 'Interpreter', 'latex');
title('Stabilizability of Linearized System vs. $x_1$', 'Interpreter', 'latex');
grid on;
