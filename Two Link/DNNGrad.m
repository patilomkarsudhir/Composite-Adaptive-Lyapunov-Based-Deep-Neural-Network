
function [Phi,Phi_prime,Xi] = DNNGrad(k,L,L_in,L_out,x,theta,act)

Phi_p=x; %Zeroth Layer Input is just x
L_vec_in=(L_in+1)*L; %Length of vectorized input layer weights
L_vec_mid=(L+1)*L; %Length of vectorized intermediate layer weights
L_vec_out=(L+1)*L_out; %Length of vectorized output layer weights
L_vec=L_vec_in+(k-1)*(L+1)*L+L_vec_out; %Total vectorized weight length

Phi_prime=zeros(L_out,L_vec); %Phi_prime matrix is initialized for the sake of computation

%Forward pass to compute Phi and store the gradients of activations in
%memory

if(k==1)
    
        vecV_p=theta(1:L_vec_in);  %Vectorized Input Layer Weight
        [Phi_p,Lambda_p,Xi_p]=LayerGrad(Phi_p,vecV_p,L,L_in,act);  %See the comments in LayerGrad Code in case you need to know further details
        V_out=unvec(theta(L_vec-L_vec_out+1:L_vec),L+1,L_out); %Unvectorized output layer weight matrix
        Phi=V_out'*[Phi_p;1]; %Output
        Phi_prime(1:L_out,1:L_vec_in)=V_out'*Lambda_p; %Jacobian w.r.t. input weights
        Phi_prime(1:L_out,L_vec_in+1:L_vec)=kron(eye(L_out),[Phi_p;1]'); %Jacobian w.r.t. output weights
        Xi=V_out'*Xi_p; %Jacobian w.r.t inputs
        
else

    for p=1:k

        if(p==1)
            vecV_p=theta(1:L_vec_in);  %Vectorized Input Layer Weight
            [Phi_p,Lambda_p,Xi_p]=LayerGrad(Phi_p,vecV_p,L,L_in,act);  %See the comments in LayerGrad Code in case you need to know further details
            Lambdas(1:L+1,1:L_vec_in)=Lambda_p; %Gradient of phi_1 w.r.t. vec(V). Biases are included.
            Xis(1:L+1,1:(L_in+1))=Xi_p; %Xi's are gradients of activation w.r.t. to inputs instead of weights
        else

            vecV_p=theta(L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid);

            [Phi_p,Lambda_p,Xi_p]=LayerGrad(Phi_p,vecV_p,L,L,act);  

            Lambdas(1:L+1,L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid)=Lambda_p;
            Xis(1:L+1,(L_in+1)+(p-2)*(L+1)+1:(L_in+1)+(p-1)*(L+1))=Xi_p;

        end
    end

    V_out=unvec(theta(L_vec-L_vec_out+1:L_vec),L+1,L_out); %Unvectorized output layer weight matrix

    Phi=V_out'*[Phi_p;1];  %Output of the DNN
    Phi_prime_out=kron(eye(L_out),[Phi_p;1]'); %Gradient at the output-layer
    Phi_prime(1:L_out,L_vec-L_vec_out+1:L_vec)=Phi_prime_out; %Store Output Layer Gradient in Phi_prime matrix

    Prod_p=eye(L_out); %Matrix initialized for iterative multiplication in the following loop

    %Backward pass to compute the products of weights and gradients of activation functions

    for p=k:-1:1

        if(p==1)

            Lambda_p=Lambdas(1:L+1,1:L_vec_in); %Used to compute Phi_prime in the next line
            Phi_prime(1:L_out,1:L_vec_in)=Prod_p*Lambda_p; %Yields the final Phi_prime gradient term
            Prod_p=Prod_p*Xis(1:L+1,1:(L_in+1)); %Iteration to compute the product of weights and gradients of activation functions

        else
            if(p==k)
                Lambda_p=Lambdas(1:L+1,L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid);
                Phi_prime(1:L_out,L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid)=Prod_p*V_out'*Lambda_p;
                Prod_p=Prod_p*V_out'*Xis(1:L+1,(L_in+1)+(p-2)*(L+1)+1:(L_in+1)+(p-1)*(L+1));  
            else
                Lambda_p=Lambdas(1:L+1,L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid);
                Phi_prime(1:L_out,L_vec_in+(p-2)*L_vec_mid+1:L_vec_in+(p-1)*L_vec_mid)=Prod_p*Lambda_p;
                Prod_p=Prod_p*Xis(1:L+1,(L_in+1)+(p-2)*(L+1)+1:(L_in+1)+(p-1)*(L+1));  

            end
        end
        
    end
    Xi=Prod_p;
end

end















