function [Phi,Lambda,Xi]= LayerGrad(x,vecV,L_out,L_in,act)


    xa=[x;1];   % Augmented x
    V=unvec(vecV,L_in+1,L_out);  % Unvectorize; convert vectorized V back to the matrix form. Dimensions of V are (L_in+1) by L_out 

if(act=='tanh')
    
    

    Phi=tanh(V'*xa);  %Output of the activation augmented. Phi is a vector with length L_out. 
%     No need to augment Phi here since it will get augmented in the next
%     iteration. We are augmenting only the input.


    Lambda=[diag(sech2(V'*xa))*kron(eye(L_out),xa');zeros(1,L_out*(L_in+1))]; %Gradient of Phi. Dimensions are (L_out+1) by L_out*(L_in+1).
    
    
    Xi=[diag(sech2(V'*xa))*V'*[eye(L_in) zeros(L_in,1);zeros(1,L_in) 0];zeros(1,L_in+1)]; %Gradient of Phi w.r.t. x. Dimensions are (L_out+1)*(L_in+1).
    
else
    
    if(act=='relu')
        
        Phi=relu(V'*xa);  %Output of the activation augmented. Phi is a vector with length L_out. 
    %     No need to augment Phi here since it will get augmented in the next
    %     iteration. We are augmenting only the input.


        Lambda=[diag(st(V'*xa))*kron(eye(L_out),xa');zeros(1,L_out*(L_in+1))]; %Gradient of Phi. Dimensions are (L_out+1) by L_out*(L_in+1).


        Xi=[diag(st(V'*xa))*V'*[eye(L_in) zeros(L_in,1);zeros(1,L_in) 0];zeros(1,L_in+1)]; %Gradient of Phi w.r.t. x. Dimensions are (L_out+1)*(L_in+1).      
        
        
    end
    
% Feel free to add more activations        
        
        
        
end

