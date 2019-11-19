function X=WEIGHTS_BIAS_OPTIMIZATION(J,X,lx,tau,ee2,CGSDLM,cgitermax)

aLtL=tau*eye(lx,lx);
A=J'*J+aLtL; 
b=J'*ee2;
    
if CGSDLM==1 % Conjugate gradient
    dX=-conjugate_gradient(A,b,cgitermax);
elseif CGSDLM==2 % Steepest descent
    dX=-steepest_descent(A,b,cgitermax);
else % Levenberg-Marquardt
    dX=-A^-1*b;
end

X=X+dX;