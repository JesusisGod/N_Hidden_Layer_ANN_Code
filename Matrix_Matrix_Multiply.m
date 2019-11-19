function WTS=Matrix_Matrix_Multiply(W,S,prev_hidden_WT,curr_hidden,prev_hidden_S)
% WT(prev_hidden_WT,curr_hidden) % W-transpose (vector or matrix)
% S (curr_hidden ,prev_hidden_S) % S (vector or matrix)
% Hence input size is: prev_hidden_WT x curr_hidden x prev_hidden_S
% NOTE: If matrix W is supplied remember to sort it to transpose indexing
% S=[-1 1 -2 1 3 5];
% W=[0.890903252535799,0.959291425205444, 0.5, 0.4, 0.3,0.2 ];
% prev_hidden_WT=2;% row of W transpose
% curr_hidden=2; % common length between W2-transpose and s22
% 
% prev_hidden_S=2; %column of S

row_WT=prev_hidden_WT;

col_S = prev_hidden_S;

common_hidden=curr_hidden;

WTS=zeros(row_WT*col_S,1);
ccc=0;

for i=1:row_WT % move along row of WT 
    for j=1:col_S
        WW=0;
        for kk=1:common_hidden % move along column % (previous hidden layer))
            WW=WW+W(common_hidden*(i-1)+kk)*S(common_hidden*(j-1)+kk); 
        end
        ccc=ccc+1;
        WTS(ccc)=WW;
    end
end