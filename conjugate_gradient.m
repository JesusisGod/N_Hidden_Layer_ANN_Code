function x=conjugate_gradient(A,b,niter)
%% Adapted from the "Numerical recipes in FORTRAN 77, Vol 1, 2nd edition"
%%             Authors: Press, W.H, Teukolsky, S. A., Vetterling, W. T., Flannery, B. P

tol=1.e-6;
err=10*tol;
row=length(b);
x=zeros(row,1);
r=b-A*x; 
z=r; 
bnrm=sqrt(sum(b.^2));
if(bnrm==0), disp('bnrm=0'); end 
iter=0;
while iter<=niter && err>tol
    iter=iter+1;
    bknum=sum(z.*r);
    if (iter==1 || mod(iter,row)==0)
        p=z;
    else        
        bk=bknum/bkden;
        p=bk*p+z;
    end
    bkden=bknum;
    z=A*p; 
    akden=sum(z.*p);
    alpha=bknum/akden;
    x=x+alpha*p;
    r=b-A*x; 
    z=r; 
    err=sqrt(sum(r.^2))/bnrm;          
end
