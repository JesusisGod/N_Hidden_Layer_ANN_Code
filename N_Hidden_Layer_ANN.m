function X=N_Hidden_Layer_ANN(PARAFILE,CGSDLM,cgitermax)

[X,lx,S,n,t,p,lp,activation_fn,aa,ma,ms,Windx,ncols,nrow,...
          nlayers,nhidden_layers,hidden_neurons,SM,...
          Cum_hidden_out,Cum_hidden,...
          tau,mulf,divf,max_epoch,stp]=ANN_PARAMETER_SET(PARAFILE);

% CGSDLM=1; %1=Conjugate gradient; 2=Steepest descent; 3=Levenberg Marquardt
% cgitermax=20;
ev=zeros(1,1);
MSE=zeros(1); % MEAN SQUARED ERROR % Performance index used in MATLAB
for nn=1:max_epoch
    for q=1:lp % Number of Patterns
        %% FEEDFORWARD
        [a2,aa]=FEEDFORWARD(p,X,aa,S,activation_fn,q,nlayers);
        %% ERROR
        e=ERROR_VECTOR(t,a2,SM,q); for ii=1:SM, ev((q-1)*SM+ii,1)=e(ii); end
        
        %% BACKPROPAGATION
        J1=BACKPROPAGATION(aa,S,X,Windx,SM,n,nlayers,nhidden_layers,ma,ms,activation_fn,...
                         hidden_neurons,Cum_hidden,Cum_hidden_out,ncols,nrow);
        J(1+(q-1)*SM:q*SM,:)=J1;
    end
       %% PERFORMANCE INDEX
    MSE(nn)=PERFORMANCE_INDEX(ev,n); 
    
    if MSE(nn)<stp
        break;
    end
       %% OPTIMIZATION
    X=WEIGHTS_BIAS_OPTIMIZATION(J,X,lx,tau,ev,CGSDLM,cgitermax);
    
       %% Levenberg-Marquardt
       if(nn>1)
        if(MSE(nn)<MSE(nn-1))
            tau=tau/divf; % Converging
        else
            tau=tau*mulf; % Diverging
        end
       end
    
end

%% Check results

%% FEEDFORWAD FROM LAYER 1 TO N
n1=zeros(1);
comp_data=zeros(1);
cuu=0;
for q=1:lp
    ccc=0;
    TTWB=0;
    a0=p(:,q);
    
    for ii=1:nlayers
        rowW8=S(ii+1);
        colW8=S(ii);
        TTW=TTWB+rowW8*colW8; % Total weight in layer ii without bias included
        TTWB=TTW+rowW8;  % Total weights and bias in layer ii
        for jj=1:rowW8 %S(ii+1) % row of W(ii)
            n1(jj)=0;
            for kk=1:colW8 %S(ii) % Column of W(ii)
                ccc=ccc+1;
                W=X(ccc);
                n1(jj)=n1(jj)+W*a0(kk);
            end

            b=X(TTW+jj); % bias
            n1(jj)=n1(jj)+b;         
        end
        ccc=TTWB;


        if activation_fn(ii)==1
            a2=purelin(n1(1:rowW8));
        elseif activation_fn(ii)==2
            a2=logsig(n1(1:rowW8));
        end       
        a0=a2; % Output becomes input into another layer
    end
    for uu=1:S(nlayers+1)
        cuu=cuu+1;
        comp_data(cuu)=a2(uu); %stores a0,a1, etc
    end
        
end
outputs=[reshape(t,S(nlayers+1)*lp,1) comp_data'];

%%
figure;
plot(MSE,'linewidth',1.5)
grid on; grid minor;
title('Mean squared error (MSE)')
xlabel('Epoch')
ylabel('Mean squared error (MSE)')
set(gca,'fontsize',14,'fontweight','bold')
axis tight

figure;
plot(log10(MSE),'linewidth',1.5)
grid on; grid minor;
title('log_{10} (Mean squared error (MSE))')
xlabel('Epoch')
ylabel('log_{10}(MSE)')
set(gca,'fontsize',14,'fontweight','bold')
axis tight

%%
figure;
% subplot(122)
plot(p,outputs(:,1),'linewidth',1.5)
hold on;
plot(p,outputs(:,2),'ro','linewidth',1.5)
grid on; grid minor;
title('Poisson ratio')
xlabel('V_p/V_s')
ylabel('\sigma')
set(gca,'fontsize',14,'fontweight','bold')
axis tight
legend('True','ANN')

