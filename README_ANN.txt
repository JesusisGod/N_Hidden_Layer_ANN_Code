The Instructions here are for running the MALAB code as a supplement to the paper entitled:
"N-hidden layer Artificial Neural Network Toolbox: MATLAB code"

The concept of artificial neural network (ANN) requires the supply of pair of input and target patterns that will be used
to train the network before producing the network weights and biases.
For example if we choose to approximate the Poisson ratio, say f(p), in equation (12) in the text, where p represents the Vp/Vs within 1.5<=p<=8.9, 
we must supply both the input p and the target f(p) for the training.

A PARAMETER text file should be prepared to facilitate the training of the network. 
(Although the code can be edited in such a way to manually input p and f(p) but this instruction is about the PARAMETER file).
Now, going back to the approximation of f(p) an example of a parameter file is given as: Vp_Vs_ratio_parameter_Fourlayer_14441	.txt

The PARAMETER file is a 10-line file given as: 

INPUT_Vp_Vs_ratio_38.txt
OUTPUT_Poisson_ratio_38.txt
WEIGHT_Vp_Vs_ratio_Fourlayers_14441.txt
1
38
4
4 4 4
1
2 2 2 1
500 1 1.0E-7 2 3

The first line contains the file name (INPUT_Vp_Vs_ratio_38.txt) for the input pattern p from 1.5 to 8.9, having 38 length
The second line contains the file name (OUTPUT_Poisson_ratio_38.txt) for the output pattern f(p) for p from 1.5 to 8.9, having 38 length
The third line is the file name for the inital weights and biases (WEIGHT_Vp_Vs_ratio_Fourlayers_14441.txt), whose length is dependent on the number of layers and neurons required for the training (see equation 4 in the text); these weight values can be generated using MATLAB rand function
The fourth line in the parameter file is the dimension of the input pattern, which is 1.
The fifth line is the number of patterns, which is 38.
The sixth line is the number of layers which is 4 in this case (which means 3 hidden layers)
The 7th line contains three numbers (equal to the number of hidden layers); and each value is the number of neurons in each hidden layer. In this case the numbers are 4 4 and 4, meaning four neurons are in the first, second and third hidden layers respectively.
The eighth line shows the number of neurons in the target output (f(p)) layer as 1.
The ninth line gives the flags for different activation function for each layer: flags 1 and 2 for purelin and logsig respectively.
The 10th line is for optimization paramter setting: the list 500 1 1.0E-7 2 3, are for 500 epoch (iterations), initial regularization term of 1, iteration stopping threshold value of 1.0E-7, and Levenberg-Marquardt multiplier and divisor, 2 and 3 respectively. 


Once the parameter file is ready, the DRIVER_N_Hidden_Layer_ANN.m should be opened and run
The main function is the N_Hidden_Layer_ANN.m which accepts three inputs namely the PARAFILE, CGSDLM flag for the type of optimization method and CGITERMAX for the number of iterations within the conjugate gradient and steepest descent algorithm per epoch.
PARAFILE is the parameter file described above.
CGSDLM=1, 2 or 3 for the conjugate gradient, steepest descent and Levenberg-Marquardt methods respectively.
CGITERMAX = number of iterations in the conjugate gradient and steepest descent per epoch.

The result from this instruction is shown in Figures 2(c) and 3(c) in the text for the log10(MSE) and Poisson approximation respectively for a 4-layer ANN with architecture of 1-4-4-4-1 using the steepest descent method.
