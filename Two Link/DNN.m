function [Phi] = DNN(k,L,L_in,L_out,x,theta,act)

Phi_p=x; %Zeroth Layer Input is just x
L_vec_in=(L_in+1)*L; %Length of vectorized input layer weights
L_vec_mid=(L+1)*L; %Length of vectorized intermediate layer weights
L_vec_out=(L+1)*L_out; %Length of vectorized output layer weights
L_vec=L_vec_in+(k-1)*(L+1)*L+L_vec_out; %Total vectorized weight length

%Forward pass to compute Phi and store the gradients of activations in
%memory

if(k==1)
    
        vecV_p=theta(1:L_vec_in);  %Vectorized Input Layer Weight
        [Phi_p]=LayerGrad(Phi_p,vecV_p,L,L_in,act);  %See the comments in LayerGrad Code in case you need to know further details
        V_out=unvec(theta(L_vec-L_vec_out+1:L_vec),L+1,L_out); %Unvectorized output layer weight matrix
        Phi=V_out'*[Phi_p;1]; %Output
        
else

    for p=1:k

        if(p==1)
            vecV_p=theta(1:L_vec_in);  %Vectorized Input Layer Weight
            [Phi_p]=LayerGrad(Phi_p,vecV_p,L,L_in,act);  %See the comments in LayerGrad Code in case you need to know further details
        else

            vecV_p=theta(L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid);

            [Phi_p]=LayerGrad(Phi_p,vecV_p,L,L,act);  

        end
    end

    V_out=unvec(theta(L_vec-L_vec_out+1:L_vec),L+1,L_out); %Unvectorized output layer weight matrix

    Phi=V_out'*[Phi_p;1];  %Output of the DNN



end















