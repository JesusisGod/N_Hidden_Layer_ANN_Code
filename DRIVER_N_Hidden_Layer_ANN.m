clear; clc; close all;
PARAFILE='Vp_Vs_ratio_parameter_Fourlayer_14441.txt';
CGSDLM=2;     %1=Conjugate gradient; 2=Steepest descent; 3=Levenberg Marquardt
CGITERMAX=20; %Number of iterations within conjugate gradient and steepest descent functions per epoch

X=N_Hidden_Layer_ANN(PARAFILE,CGSDLM,CGITERMAX);