# ML_Project_Spring_2021

## Naive Bayes
- No classes were implemented for the naive bayes implementation, only two functions: naive_bayes_train and naive_bayes_predict
- The notebook called digit_recognition_nb.ipynb loads all of the training and test data and formats it for the above functions
- There are comments in the functions and the notebook describing what each block of code does
- after training, each mean image is plotted
- After testing, teh overall accuracy is computed, followed by the accuracy for each specific digit
- scipy.stats is required to run this implementation

## Support Vector Machine
- The SVM code is all inside of the svm folder.
- svm_digit_recognition_testing is used for testing the different kernels and hyperparameters. It uses 10 batches of 100 training examples and 1000 test examples.
- svm_digit_recognition_final is the final model developed based off the test results. It uses 2,000 training examples and 10,000 testing examples.
- Libraries used for SVM:
    - numpy
    - pandas
    - quadprog

## Convolutional Neural Net
- All functionality for the CNN is contained in cnn.py. The following libraries are used in cnn.py:
    - numpy
    - pandas (used for importing from CSV)
    - torch (used to build the CNN)
    - sys (for command line argument checking)
- There are 2 options when running cnn.py, you can run it to train and test over 20 epochs or you can run it to load and test the saved model that was found to have the highest accuracy on the test dataset (99.325%)
    - Training and testing over 20 epochs can be done by simple calling cnn.py without any command line arguments ("python cnn.py"). A summary bar graph showing loss and accuracy from every epoch will be displayed after testing is concluded
    - Loading and testing the saved model can be done by including the argument --best ("python cnn.py --best")
- The CNN will attempt to use CUDA cores if present on the host machine. Typical runtime on the primary test
machine was about 2 minutes using CUDA cores when training and testing over 20 epochs. It's much longer (40+ minutes) if there are no CUDA cores available. These times are approximate and will vary with different hardware, for reference the primary test machine had an 8 core/16 thread CPU and a GPU with approximately 2,400 CUDA cores.
