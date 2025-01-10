1. Training Dataset Generation
The "TrainingData_generation.m" was used to generate the simulated dataset through the FDTD method. To run this file, please modify the data save path accordingly.

2. Network Training
The "main_train.py" provided codes to train the network which incorporates U-Net and Conv-LSTM. Similarly, please modify the training dataset path and the trained model save path accordingly.

3. Testing
The "main_test.py" provided codes to validate the performance of the network on the simulated dataset. Similarly, please modify the testing dataset path.

*** Notice ***
To test the FDTDNet on real MRE data, you need to transform your MRE data into the same type of the simulated data (including matrix size 256*256, the number of phase 8, data type ".mat"). After that, reference the file "main_test.py", you could obtain the stiffness estimation of your MRE data. If you have any problems in your implementation, please feel free to ask me.