function J=BACKPROPAGATION(aa,S,X,Windx,SM,n,nlayers,nhidden_layers,ma,ms,activation_fn,...
                         hidden_neurons,Cum_hidden,Cum_hidden_out,ncols,nrow)

%% Store sensitivities for all hidden layers (i.e excluding input and output layers), the output layer sensitivity will be computed in subsequent loop
         ccs=0;
         F1=zeros(1);
         F2=zeros(SM,1);
         J=zeros(SM,n); % Per pattern
         for ii=1:nhidden_layers
            for iii=ma(ii+1,1):ma(ii+1,2) %hi_n
                ccs=ccs+1;
                a1ii=aa(iii);
                
                if activation_fn(ii)==1
                    F1(ccs)=1;
                elseif activation_fn(ii)==2
                    F1(ccs)=(1-a1ii)*a1ii;
                end
                
            end   
         end
         
        for k=1:SM % Run through number of outputs per pattern         
    %% Backpropagate sensitivity (error) starting from last layer            
            % Last layer sensitivity of pure linear function
            for ii=1:SM
                F2(ii,1)=0;
            end   
            
            if activation_fn(nlayers)==1
                F2(k,1)=1; % Only the output involved will be non-zero
            elseif activation_fn(nlayers)==2
                ii=ma(nlayers+1,1)+k-1;
                a1ii=aa(ii);
                F2(k,1)=(1-a1ii)*a1ii; % Only the output involved will be non-zero
            end
            s22=-F2;            
            cch=0;
            for ii=S(nlayers+1):-1:1
                
                iii=Cum_hidden_out(nlayers)-cch;
                SS(iii,1)=s22(ii); %[s11;s22];
                cch=cch+1;
                iii_keep=iii;
            end
            % The hidden layer sensitivity is then computed by backpropagating the
            % sensitivity from the last layer
            for ii=1:nhidden_layers
                Sindx=nlayers+1-ii+1; % Backpropagation starts from the last layer
                curr_hidden=S(Sindx); % row of W; row of S; but col for WT
                prev_hidden_WT=S(Sindx-1); % col of W; row of WT
                prev_hidden_s22=1; % col of s22
                an=2*nlayers-1-2*(ii-1);
                WW=X(Windx(an,1):Windx(an,2),1); % Note that WW size has to be defined as max of weight length in FORTRAN
                WT=Wtranspose(WW,curr_hidden,prev_hidden_WT); % row of WW=curr_hidden; col of WW=prev_hidden
                WTS2=Matrix_Matrix_Multiply(WT,s22,prev_hidden_WT,curr_hidden,prev_hidden_s22);
                cch1=0;
                jjj=Cum_hidden(nhidden_layers-ii+1); % starting index in F1
                
                for hn=hidden_neurons(nhidden_layers-ii+1):-1:1 %S(nlayers-ii+1):-1:1
                    cch1=cch1+1;
                    s11=F1(jjj)*WTS2(hn); %F1*W2'*s22;
                    iii_keep=iii_keep-1;
                    SS(iii_keep)=s11;
                    s22(hn)=s11; % To re-initialize s22 for next hidden layer iteration
                    jjj=jjj-1;
                end
            end     
            h=k; % for individual subroutine
                L=0;
                    for m=1:nlayers %:-1:1
                        s=SS(ms(m,1):ms(m,2));
                        a=aa(ma(m,1):ma(m,2));
                        for ib=1:2 % W for ib=1; bias for ib=2
                            if(ib==1),ncol=ncols(m); else ncol=1; end
                            for j=1:ncol                                 
                                for i=1:nrow(m) % Number of elements in each bias                         
                                    L=L+1;
                                    if(ib==1)
                                        J(h,L)=s(i)*a(j); %SS(m,i,h)*aa(m-1,j,q);
                                    else
                                        J(h,L)=s(i); %SS(m,i,h)*aa(m-1,j,q);
                                    end
                                end
                            end
                        end
                    end

        end