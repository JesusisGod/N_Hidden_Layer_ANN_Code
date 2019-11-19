function [a2,aa]=FEEDFORWARD(p,X,aa,S,activation_fn,q,nlayers)
% q=1;
ccc=0;
%TTW=0;
TTWB=0;
a0=p(:,q);
cuu=0;
n1=zeros(1,1);
for uu=1:S(1) % For input layer
    cuu=cuu+1;
    aa(cuu)=a0(uu); %stores a0,a1, etc
end
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

    for uu=1:S(ii+1)
        cuu=cuu+1;
        aa(cuu)=a2(uu); %stores a0,a1, etc
    end
    a0=a2; % Output becomes input into another layer
end
            