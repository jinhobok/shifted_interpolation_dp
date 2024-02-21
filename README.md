# shifted_interpolation_dp
A repository for reproducing the numerical results in the paper "Shifted Interpolation for Differential Privacy"

# Files
+ ` gdp.py ` : utils for calculating Gaussian tradeoff functions
+ ` plot_intro.py ` : code for generating Figure 1 (exemplary numerics on NoisyGD) and Figure 2 (Illustration of f-DP and GDP)
+ ` plot_nsgd_fdp.py ` : code for generating Figure 10 (numerics on NoisySGD)
+ ` plot_nsgd_opt.py ` : code for generating Figure 4 (one-step optimality of privacy bound of NoisySGD)
+ ` plot_table_ncgd_fdp.py ` : code for generating Figure 8-9 and Table 9-12 (numerics on NoisyCGD)
+ ` plot_table_ngd_fdp.py ` : code for generating Figure 6-7 and Table 7-8 (numerics on NoisyGD)
+ ` prv_SymmPoissonSubsampledGaussianMechanism.py ` : code for calculating privacy of subsampled Gaussian mechanism (see Appendix D.5)
+ ` table_acc_lr.py ` : code for generating Table 2, 5-6 (train and test accuracy of the experiment on regularized logistic regression)
+ ` table_privacy_lr.py ` : code for generating Table 1, 3-4 (privacy of the experiment on regularized logistic regression)
+ ` MNISTdata.hdf5 ` : MNIST data
+ ` requirements.txt ` : required packages

# Remarks
+ The formats of tables and figures obtained from the codes may have been modified for better presentation in the paper.

# Reference
+ Regularized logistic regression: https://github.com/yawen-d/Logistic-Regression-on-MNIST-with-NumPy-from-Scratch
+ PRV framework: https://github.com/microsoft/prv_accountant
