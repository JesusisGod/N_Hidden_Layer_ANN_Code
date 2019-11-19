function [X,lx,S,n,t,p,lp,activation_fn,aa,ma,ms,Windx,ncols,nrow,...
          nlayers,nhidden_layers,hidden_neurons,SM,...
          Cum_hidden_out,Cum_hidden,...
          tau,mulf,divf,niter,stp]=ANN_PARAMETER_SET(PARAFILE)
      
 fid=fopen(PARAFILE);
 INPUT_PATTERN_T=textscan(fid,'%s',1,'Delimiter','|');
 OUTPUT_PATTERN_T=textscan(fid,'%s',1,'Delimiter','|');
 WEIGHTFILE_T=textscan(fid,'%s',1,'Delimiter','|');
 D_T=textscan(fid,'%d',1);
 N_patterns_T=textscan(fid,'%d',1);
 nlayers_T=textscan(fid,'%d',1);

 
 INPUT_PATTERN=char(INPUT_PATTERN_T{1,1});
 OUTPUT_PATTERN=char(OUTPUT_PATTERN_T{1,1});
 WEIGHTFILE=char(WEIGHTFILE_T{1,1});
 D=D_T{1,1};                   % Dimension of input= R in the literature
 N_patterns=N_patterns_T{1,1}; % Number of patterns
 
 nlayers=nlayers_T{1,1};       % Number of layers
 nhidden_layers=nlayers-1;     % Number of hidden layers
 hidden_neurons_T=textscan(fid,'%d',nhidden_layers);
 output_neurons_T=textscan(fid,'%d',1);
 activation_fn_T=textscan(fid,'%d',nlayers);
 inversn_para_T=textscan(fid,'%d %f %f %f %f');
 
 hidden_neurons=hidden_neurons_T{1,1};
 output_neurons=output_neurons_T{1,1};
 activation_fn=activation_fn_T{1,1};
 niter=inversn_para_T{1,1};
 tau=inversn_para_T{1,2};
 stp=inversn_para_T{1,3};
 mulf=inversn_para_T{1,4};
 divf=inversn_para_T{1,5};
 
 
input_pattern_DB=load(INPUT_PATTERN); % Input pattern database
output_pattern_DB=load(OUTPUT_PATTERN); % output
WEIGHTS_BIASES=load(WEIGHTFILE);
%  hidden_neurons=
%% I observe that the nlayers <= D (dimension of input); but check literature for confirmation
%%
% % input_pattern_DB=load('INPUT_two_4D_patterns_two_outputs.txt'); % Input pattern database
% % output_pattern_DB=load('OUTPUT_two_4D_patterns_two_outputs.txt'); % output
% % nlayers=3; % 2 layers= 1 hidden layer + 1 Output layer
% % nhidden_layers=nlayers-1; hidden_neurons=1*ones(nhidden_layers,1); 
% % hidden_neurons(2)=2; output_neurons=2; %
% % D=4; % Dimension of input
% % N_patterns=2; % equivalent to lp=number of patterns for training
%%
% input_pattern_DB=load('INPUT_quadratic2_three_inputs.txt'); % Input pattern database
% output_pattern_DB=load('OUTPUT_quadratic2_one_outputs.txt'); % output
% nlayers=3; % 2 layers= 1 hidden layer + 1 Output layer
% nhidden_layers=nlayers-1; hidden_neurons=1*ones(nhidden_layers,1); 
% hidden_neurons(2)=2; output_neurons=1; %
% D=3; % Dimension of input
% N_patterns=3; % equivalent to lp=number of patterns for training
%%

% % input_pattern_DB=load('INPUT_sine_41_inputs.txt'); % Input pattern database
% % output_pattern_DB=load('OUTPUT_sine_41_outputs.txt'); % output
% % nlayers=55; % 2 layers= 1 hidden layer + 1 Output layer
% % nhidden_layers=nlayers-1; hidden_neurons=1*ones(nhidden_layers,1); 
% % hidden_neurons(2)=2; output_neurons=41; %
% % D=41; % Dimension of input
% % N_patterns=1; % equivalent to lp=number of patterns for training

%% [lp,tau,mulf,divf,niter,stp]
SM=output_neurons;
lp=N_patterns;
S0=D; % equivalent to R = in_n
% tau=1e-0;
% mulf=2;
% divf=1.05;
% niter=100; %000;
% stp=1e-7;

p=zeros(S0,N_patterns);             % Input patterns
t=zeros(SM,N_patterns); % target

for i=1:N_patterns
    p(:,i)=input_pattern_DB (1+(i-1)*S0:i*S0);
    t(:,i)=output_pattern_DB(1+(i-1)*SM:i*SM);
end


%%
S=zeros(1,nlayers+1);
S(1,1)=S0;

for i=1:nhidden_layers
    S(1,i+1)=hidden_neurons(i);
end
% 
S(1,nlayers+1)=SM;
Tot_S=sum(S);
Cum_S=cumsum(S);
Cum_hidden_out=cumsum(S(2:end));
Cum_hidden=cumsum(S(2:nlayers));
%%
n=0;
for i=1:nlayers
    n=n+S(i+1)*(S(i)+1);
end
Windx=zeros(2*nlayers,2); % Weight and bias indexing
Windx(1,1)=1;
cc=0;
W=0;
for i=1:nlayers
    
    W=W+S(i+1)*S(i); % Weight section
    cc=cc+1;
    Windx(cc,2)=W;
    
    W=W+S(i+1); % Bias section
    cc=cc+1;
    Windx(cc,2)=W;
end

for i=1:2*nlayers-1
    Windx(i+1,1)=Windx(i,2)+1;    
end

%% Pre-defined indexes
ma=zeros(nlayers+1,2);
ms=zeros(nlayers,2);

ncols=zeros(nlayers,1);
nrow=ncols;

ma1=1;
ma2=0; %S(1); % starting from input S0


ms1=1;
ms2=0;

for i=1:nlayers  % Since ma2 starts at S(1)
    ma2=ma2+S(i); 
    ms2=ms2+S(i+1); % starting from the first hidden layer
    
    ma(i,1:2)=[ma1 ma2];
    ms(i,1:2)=[ms1 ms2];
    
    ms1=ms2+1; %ms2=ms2+S(i+2);    
    ma1=ma2+1;   
    
    ncols(i)=S(i);
    nrow(i)=S(i+1);
end

ma2=ma2+S(nlayers+1);
ma(nlayers+1,1:2)=[ma1 ma2]; % For storing values in aa



% WEIGHTS_BIASES=rand(n,1);
% for hidden layer=1

% % if nlayers==2
% % WEIGHTS_BIASES=[0.350727103576883;0.939001561999887;0.875942811492984;0.550156342898422;...
% %                 0.622475086001228;...
% %                 0.587044704531417;0.207742292733028;...
% %                 0.301246330279491;0.470923348517591];
% % 
% % end
% % 
% % if nlayers==3 && hidden_neurons(2)==1
% % % for hidden layer=2
% % WEIGHTS_BIASES=[0.392227019534168;0.171186687811562;0.0318328463774207;0.0461713906311539;...
% %                 0.655477890177557;...
% %                 0.706046088019609;...
% %                 0.276922984960890;...
% %                 0.0971317812358475;0.823457828327293;... %0.694828622975817;
% %                 0.381558457093008;0.765516788149002];
% % elseif nlayers==3 && hidden_neurons(2)==2
% % %     WEIGHTS_BIASES=[0.743132468124916;0.392227019534168;0.655477890177557;0.171186687811562;...
% % %                     0.706046088019609;...
% % %                     0.0318328463774207;0.276922984960890;...
% % %                     0.0461713906311539;0.0971317812358475;...
% % %                     0.823457828327293;0.317099480060861;0.694828622975817;0.950222048838355;...
% % %                     0.0344460805029088;0.438744359656398];
% %     WEIGHTS_BIASES=[0.669175304534394;0.190433267179954;0.368916546063895; 0.460725937260412;...
% %                     0.981637950970750;...
% %                     0.156404952226563;0.855522805845911;...
% %                     0.644764536870088;0.376272210278832;...
% %                     0.190923695236303;0.482022061031856;0.428252992979386;0.120611613297162;...
% %                     0.589507484695059;0.226187679752676];
% %                     
% % end
%% QUADRATIC2; NLAYERS=3; OUTPUT=1
% WEIGHTS_BIASES=[0.679702676853675;0.655098003973841;0.162611735194631;0.118997681558377;0.498364051982143;0.959743958516081;0.340385726666133;0.585267750979777;0.223811939491137;0.751267059305653;0.255095115459269];

%%
X=WEIGHTS_BIASES;
lx=length(X);
% activation_fn=2*ones(1,nlayers); %[2 2 1];
% activation_fn(1,nlayers)=1;

aa=zeros(Tot_S,1);