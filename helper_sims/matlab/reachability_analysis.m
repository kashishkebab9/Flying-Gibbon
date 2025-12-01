% Define symbolic variables
syms x1 x2 x3 x4 r m I g real     % State variables and parameters
syms F_l F_r tau_a real           % Inputs

m_val = 1.0;
r_val = 0.2;
I_val = 1.0;
g_val = 9.81;

% State and input vectors
x = [x1; x2; x3; x4];
u = [F_l; F_r; tau_a];

% System dynamics
f = [x2;
     -(g*sin(x1)) - (1/m)*(sin(x1 - x3))*(F_l + F_r);
     x4;
     -r*F_l + r*F_r + I*tau_a];

% Compute Jacobians for linearization
A_sym = jacobian(f, x);  % ∂f/∂x
B_sym = jacobian(f, u);  % ∂f/∂u

% Display A and B for symbolic inspection
disp('A matrix:')
pretty(simplify(A_sym))

disp('B matrix:')
pretty(simplify(B_sym))

% Compute controllability matrix
C = [B_sym, A_sym*B_sym, A_sym^2*B_sym, A_sym^3*B_sym];

% Simplify for readability
C_simplified = simplify(C);

disp('Controllability Matrix:')
pretty(C_simplified)
