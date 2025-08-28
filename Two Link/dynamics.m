function [f,g] = dynamics(X)



q=X(1:2);
qdot=X(3:4);

p1 = 3.473;
p2 = 0.196;
p3 = 0.242;

fd1 = 5.3;
fd2 = 1.1;

M = [p1+2*p3*cos(q(2)), p2+p3*cos(q(2)); p2+p3*cos(q(2)), p2];
Vm = [-p3*sin(q(2))*qdot(2), -p3*sin(q(2))*(qdot(1)+qdot(2)); p3*sin(q(2))*qdot(1), 0];
Fd = [fd1, 0; 0, fd2];

Minv=inv(M);


f=-(Vm*qdot+Fd*qdot);
g=Minv;

