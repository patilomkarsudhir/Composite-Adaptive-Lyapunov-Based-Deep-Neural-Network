function [F_error_rms]=DNN_eval(k,L,L_in,L_out,theta,X_rand)

N=length(X_rand(1,1,:));
F_error=zeros(L_out,N);
for i=1:N
    Xi=X_rand(:,:,i);
    X(:,i)=Xi;
    [Phi] = DNN(k,L,L_in,L_out,Xi,theta,'tanh');
    [f]=two_link(Xi);
    F_error(:,i)=f-Phi;
end

F_error_rms=mean(vecnorm(F_error));