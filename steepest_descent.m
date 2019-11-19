function x=steepest_descent(A,b,niter)
row=length(b);
x=zeros(row,1);
r=b-A*x; 
for iter=1:niter
    p=r;
    anum=p'*p;
    aden=p'*A*p;
    alpha=anum/aden;
    x=x+alpha*p;
    r=b-A*x;   
end