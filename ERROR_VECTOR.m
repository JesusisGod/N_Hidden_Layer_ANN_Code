function e=ERROR_VECTOR(t,a2,SM,q)
%% Error
e=zeros(1);
for ii=1:SM
    e(ii)=t(ii,q)-a2(ii);    
end