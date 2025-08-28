function [f,g,J] = dynamics(X)
% Dynamics of Unmanned Underwater Vehicle
% (see Nonlinear RISE-Based Control of an Autonomous Underwater Vehicle, 2014 for model dynamics and Quaternion Feedback Regulation of Underwater Vehicles, 1994 for the numbers and matrices) 
% X      - state vector [eta; eta_dot]
% f      - dynamics vector
% g      - control gain matrix
% J      - Jacobian matrix

eta = X(1:6);
eta_dot = X(7:12);

% Variables
x = eta(1);
y = eta(2);
z = eta(3);
phi = eta(4);
theta = eta(5);
psi = eta(6);

% Trigonometry shorthand
cpsi = cos(psi);
spsi = sin(psi);
cphi = cos(phi);
sphi = sin(phi);
ctheta = cos(theta);
stheta = sin(theta); 
ttheta = tan(theta);

% Jacobian J
J1 = [cpsi*ctheta, -spsi*cphi + cpsi*stheta*sphi, spsi*sphi + cpsi*cphi*stheta;...
    spsi*ctheta, cpsi*cphi + sphi*stheta*spsi, -cpsi*sphi + stheta*spsi*cphi;...
    -stheta, ctheta*sphi, ctheta*cphi];

J2 = [1, sphi*ttheta, cphi*ttheta;...
    0, cphi, -sphi;...
    0, (sphi/ctheta), (cphi/ctheta)];

J = [J1, zeros(3); zeros(3), J2];

% Calculate J_dot analytically
J1_dot = [
    -spsi*ctheta*eta_dot(6) - cpsi*stheta*eta_dot(5), -cphi*spsi*eta_dot(6) + cpsi*stheta*sphi*eta_dot(6) + cpsi*cphi*stheta*eta_dot(5),  cphi*spsi*eta_dot(6) + cpsi*stheta*cphi*eta_dot(6) + cpsi*sphi*stheta*eta_dot(5);
    cpsi*ctheta*eta_dot(6) - spsi*stheta*eta_dot(5), -cphi*cpsi*eta_dot(6) - spsi*stheta*sphi*eta_dot(6) + sphi*cpsi*stheta*eta_dot(5), -cphi*cpsi*eta_dot(6) + sphi*spsi*stheta*eta_dot(6) + sphi*stheta*cpsi*eta_dot(5);
    -ctheta*eta_dot(5), ctheta*sphi*eta_dot(5), ctheta*cphi*eta_dot(5)
];

J2_dot = [
    0, cphi*ttheta*eta_dot(4) + ttheta*sphi*eta_dot(5), -ttheta*sphi*eta_dot(4) + cphi*ttheta*eta_dot(5);
    0, -sphi*eta_dot(4), -cphi*eta_dot(4);
    0, (cphi*eta_dot(4)*ctheta - sphi*stheta*eta_dot(5))/(ctheta^2), -(sphi*eta_dot(4)*ctheta + cphi*stheta*eta_dot(5))/(ctheta^2)
];

J_dot = [J1_dot, zeros(3); zeros(3), J2_dot];

% Determine body-fixed velocities
nu = J \ eta_dot;

% Parameters in UUV inertial frame
m = [215, 265, 265, 40, 80, 80];
d1 = [70, 100, 200, 30, 50, 50];
d2 = [100, 200, 50, 50, 100, 100];
ddiag = [d1(1) + d2(1) * abs(nu(1)), d1(2) + d2(2) * abs(nu(2)), d1(3) + d2(3) * abs(nu(3)),...
    d1(4) + d2(4) * abs(nu(4)), d1(5) + d2(5) * abs(nu(5)), d1(6) + d2(6) * abs(nu(6))];

M = diag(m);
C = [0, 0, 0, 0, m(3) * nu(3), -m(2) * nu(2);
    0, 0, 0, -m(3) * nu(3), 0, m(1) * nu(1);
    0, 0, 0, m(2) * nu(2), m(1) * nu(1), 0;
    0, m(3) * nu(3), -m(2) * nu(2), 0, m(6) * nu(6), -m(5) * nu(5);
    -m(3) * nu(3), 0, m(1) * nu(1), -m(6) * nu(6), 0, m(4) * nu(4);
    m(2) * nu(2), -m(1) * nu(1), 0, m(5) * nu(5), -m(4) * nu(4), 0];
D = diag(ddiag);
g = zeros(6, 1);

% Parameters in earth inertial frame
M_bar = inv(J)' * M * inv(J);
C_bar = inv(J)' * (C - M * (J \ J_dot)) * inv(J);
D_bar = inv(J)' * D * inv(J);
g_bar = inv(J)' * g;

f = M_bar \ (-C_bar * eta_dot - D_bar * eta_dot + g_bar);
g = M_bar;
end
