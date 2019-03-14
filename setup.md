# Getting Started

First, install git and clone the current repository

```
sudo yum install git
git clone current-repository
```

## Prerequisites

For this program we will use python 2.7.14 with packages:

```
numpy 1.13.3
scipy 0.19.1
pandas 0.20.3
scikit-learn 0.19.1
```

## Installing 

Install python 2.7.14 and pip:

```
https://realpython.com/installing-python/
```

After python is installed run in command line
```
pip install numpy==1.13.3 # install numpy
pip install scipy==0.19.1
pip install pandas==0.20.3
pip install scikit-learn==0.19.1

```

## Running the program

Once the enviornment is set, from the command line do:

```
cd ridge
python main.py
```

## Input 
Current setup runs a dual ridge regression model with Gausian kernel (and range of possible sigma terms for the kernel). Regulariztion values for the loss function is also specified at main.py.  
Please adjust main.py if primal ridge regression and/or different regularization values are desired. 

## Output
Currently the file main.py will only print in the end of the process
the best loss values achieved by the model for training, cross-validation and testing phases. 
Please adjust main.py if writing the results to a file is desired.