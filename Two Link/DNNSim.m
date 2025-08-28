function [e,DNN_error,u_out,theta_out,F_error_rms,ftilde,time] = DNNSim(Method,k,L,n,theta_init,step_size,simtime,X_init,kr,kf,k_theta,gamma_init,alpha_1,alpha_2,alpha_3,beta_0,X_rand)

L_in=2*n;

L_out=n;

L_vec_in=(L_in+1)*L; %Length of vectorized input layer weights
L_vec_mid=(L+1)*L; %Length of vectorized intermediate layer weights
L_vec_out=(L+1)*L_out; %Length of vectorized output layer weights
L_vec=L_vec_in+(k-1)*L_vec_mid+L_vec_out; %Total vectorized weight length

time_length=simtime/step_size;

X=X_init;
theta=theta_init;
Gamma=gamma_init;

kappa_0=2*max(abs(eig(gamma_init)));

time=zeros(1,time_length);
e=zeros(n,time_length);
ftilde=zeros(n,time_length);
DNN_error=zeros(n,time_length);
u_out=zeros(n,time_length);
theta_out=zeros(L_vec,time_length);
F_error_rms=zeros(n,time_length);


act='tanh';

for i=1:time_length
    
    
       
    t=(i-1)*step_size;       %Time   
    time(i)=t;
    
    Xi=X(:,i);  
    xi=X(1:2,i);               %State
    xi_dot=X(3:4,i);
    


    xdi=0.5*0.5*exp(-sin(t))*[sin(t);cos(t)];  %Desired Trajectory
    xdi_dot=0.5*(0.5*exp(-sin(t))*[cos(t);-sin(t)]-0.5*exp(-sin(t))*cos(t)*[sin(t);cos(t)]);
    xdi_ddot=0.5*(-0.5*exp(-sin(t))*cos(t)*[cos(t);-sin(t)]-0.5*exp(-sin(t))*[sin(t);cos(t)]+0.5*exp(-sin(t))*(cos(t)^2)*[sin(t);cos(t)]+0.5*exp(-sin(t))*sin(t)*[sin(t);cos(t)]-0.5*exp(-sin(t))*cos(t)*[cos(t);-sin(t)]);
    
    ei=xi-xdi;
    e(:,i)=ei;
    
    ei_dot=xi_dot-xdi_dot;
    
    ri=ei_dot+alpha_1*ei;
    

    

    [Phi,Phi_prime] = DNNGrad(k,L,L_in,L_out,Xi,theta,act);

    
    if(i==1)
        rhat_i=ri;
        fhat_i=zeros(2,1);
        fhat_int=zeros(2,1);
    end    
    
    
    E_i=fhat_i-Phi;
    
    
    if(Method=="Traditional")
        thetadot=-k_theta*Gamma*theta+Gamma*Phi_prime'*ri;
    else
        beta=beta_0*(1-norm(Gamma)/kappa_0);
        Gammadot=beta*Gamma-Gamma*(Phi_prime'*Phi_prime)*Gamma;
        Gamma=Gamma+step_size*Gammadot;
        thetadot=-k_theta*Gamma*theta+Gamma*Phi_prime'*(ri+alpha_3*E_i);
    end
    
    theta=theta+step_size*thetadot;   
    
    
    [f,g]=two_link(Xi);
    u=pinv(g)*(xdi_ddot-Phi-(alpha_1+kr)*ri+(alpha_1^2-1)*ei);


    Xdot=[xi_dot;f]+[zeros(2,1);g*u];
    X(:,i+1)=Xi+step_size*Xdot;
    
    rhatdot=g*u-xdi_ddot+fhat_i+alpha_2*(ri-rhat_i)+alpha_1*(ri-alpha_1*ei);
    fhat_int_dot=(kf*alpha_2+1)*(ri-rhat_i);
    
    rhat_i=rhat_i+step_size*rhatdot;
    fhat_int=fhat_int+step_size*fhat_int_dot;
    
    fhat_i=kf*(ri-rhat_i)+fhat_int;
    
    ftilde(:,i)=f-fhat_i;
    
    DNN_error(:,i)=f-Phi;
    u_out(:,i)=u;

    theta_out(:,i)=theta;
    e(:,i)=ei;
    
    F_error_rms(:,i)=DNN_eval(k,L,L_in,L_out,theta,X_rand);
    
        
end


