function QQ=update_state_action_valuePG(Q,o,a,r,alpha,rho)
% modified the function by ezizizi (2023)
%QQ=update_state_action_value(Q,O(n),a,r,r0,alpha,rho);

% QQ corresponds the state-action value vector at step n+1 (updated)
% Q corresponds the state-action value vectorr at step n (old)

% This function update state-action values

QQ=Q;

if o==0
    rho_t=[rho;1-rho]; % probabilities of identifying the current state as dark and light respectively, given that the current state is dark 
elseif o==1
    rho_t=[1-rho;rho]; % probabilities of identifying the current state as dark and light respectively, given that the current state is light 
end

if (r==0)
    
%     if a == 1
%         aNew = a-1;
%     else
%         aNew = a;
%     end

    aNew = a;
    for oo=1:2

        % for action a (a=0='dark' response, a=1='light' response; we use a+1 because matrices in matlab cannot start from element 0)
        QQ(oo,aNew+1) = Q(oo,aNew+1) + alpha * ( ( (1- Q(oo,aNew+1))*rho_t(oo) ) / ( sum( (1-Q(:,aNew+1)).*rho_t ) ) ) * (0- Q(oo,aNew+1)); % update the value for the current state
%         if a == 0
%             aNew = 1;
%         else
%             aNew = 0;
%         end
    end 

elseif (r==1)

    aNew = a;
    for oo=1:2
        % for action a (a=0='dark' response, a=1='light' response; we use a+1 because matrices in matlab cannot start from element 0)
        QQ(oo,aNew+1) = Q(oo,aNew+1) + alpha * ( ( Q(oo,aNew+1)*rho_t(oo) ) / ( sum( Q(:,aNew+1).*rho_t ) ) ) * (r- Q(oo,aNew+1)); % update the value for the current state
%         if a == 0
%             aNew = 1;
%         else
%             aNew = 0;
%         end
    end
       
end


end