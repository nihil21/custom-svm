# custom-svm
Custom implementation of Support Vector Machines using Python and NumPy, as part of the Combinatorial Decision Making and Optimization university course (Master in Artificial Intelligence, Alma Mater Studiorum - University of Bologna).

### Authors
[Mattia Orlandi](https://github.com/nihil21)     
[Lorenzo Mario Amorosa](https://github.com/Lostefra)     

### Credits
[Autore, Titolo](https://static1.squarespace.com/static/58851af9ebbd1a30e98fb283/t/58902fbae4fcb5398aeb7505/1485844411772/SVM+Explained.pdf)     
[Autore, Titolo](http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/stfhtmlnode64.html)     

### Design and Implementation: Overview

The repository is structured in the following way:
 - the module [`custom-svm/svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py) contains the implementation of SVM for binary classification, with support to kernel functions and soft margin.  
 - the module [`custom-svm/multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/multiclass_svm.py) contains the implementation of SVM for multiclass classification.
 - the notebook [`custom-svm/svm_usecase.ipynb`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm_usecase.ipynb) shows the usage of the SVM for many different purposes.
 - the package [`custom-svm/data`](https://github.com/nihil21/custom-svm/tree/master/custom-svm/data) contains generators and datasets. 

### Lagrangian Formulation of the SVM

The Lagrangian problem for SVM formulated in its dual form:

![LaTeX image not found :(](res/dual.gif?raw=true)

subject to:   

![LaTeX image not found :(](res/const1.gif?raw=true)

![LaTeX image not found :(](res/const2.gif?raw=true)    

It is a quadratic optimization problem that can be solved using the quadratic library `cvxopt` in python, so it is necessary to match the solver's API which, according to the documentation, is of the form:

![LaTeX image not found :(](res/cvxopt_sign.gif?raw=true)

subject to:  

![LaTeX image not found :(](res/const3.gif?raw=true)   

![LaTeX image not found :(](res/const4.gif?raw=true)

Let **H** be a matrix such that H<sub>i,j</sub> = y<sub>i</sub> y<sub>j</sub> **x<sub>i</sub> x<sub>j</sub>** , then the function to optimize becomes:

![LaTeX image not found :(](res/dual_h.gif?raw=true)

We then convert the sums into vector form and multiply both the objective and the constraint by −1, which turns this into a minimization problem and reverses the inequality in constaints. The optimization problem can be written as:

![LaTeX image not found :(](res/dual_h2.gif?raw=true)

subject to:  

![LaTeX image not found :(](res/const5.gif?raw=true)   

![LaTeX image not found :(](res/const6.gif?raw=true)    

It is now necessary to convert the numpy arrays that express the optimization problem accordingly to `cvxopt` format. Using the same notation as in the documentation, this gives:  
 - **P**:=**H** a matrix of size m×m
 - **q**:=**−1⃗**  a vector of size m×1
 - **G**:=−diag[1] a diagonal matrix of -1s of size m×m
 - **h**:=**0⃗**  a vector of zeros of size m×1
 - **A**:=**y** the label vector of size m×1
 - b:=0 a scalar

\[...\]

### SVM for Multiclass Classification

The module [`multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/multiclass_svm.py) contains the implementation of Support Vector Machine for multi-classification purposes based on **one-vs-one strategy**.  
It offers full support to **kernel functions** and **soft margin**, in fact the signature of its `__init__` method is the same of the binary `SVM`.    
Given N different classes to classify, the algorithm provides N*(N-1)/2 SVM binary classifiers from the module [`svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py).   
**Each classifier** is **trained** to correctly classify **2 of the N** given **classes**. In the training process there are used only the entries in the dataset to which it corresponds a label of the 2 classes.   
Given an unseen example, the **prediction** of the class is computed deploying a **voting schema** among the binary `SVM` classifiers.   
The voting process is based on the standard `predict` function for binary `SVM` classifiers, so the tested entry is assigned to the class which wins the highest number of binary comparisons. In addition, it is available a mechanism to **counteract** the possible risk of **draw** in voting, based on the raw values predicted by the binary classifiers before the application of 'sign' function.

-----------------------------------------------
link latex generator: https://www.codecogs.com/latex/eqneditor.php

raw formula in order:

\max_{\lambda}\, F(\boldsymbol{\lambda}) = \sum\limits_{i=1}^{n}\lambda_i-\frac{1}{2}\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{n}\lambda_i\lambda_j\, y_i\, y_j< \mathbf{\, x_i\, x_j} > 

\lambda_i \geq 0,\: i= 1\, ...\ n

\min_{x}\, F(\boldsymbol{x}) = \frac{1}{2}\boldsymbol{x}^T\mathbf{P}\boldsymbol{x}\, +\, \boldsymbol{q}^T\boldsymbol{x}

\boldsymbol{G x} \leq  \boldsymbol{h}

\boldsymbol{A x} =  \boldsymbol{b}

H_i_,_j\, =\, y_i\, y_j\, < \mathbf{\, x_i\, x_j} >

\max_{\lambda}\, F(\boldsymbol{\lambda}) = \sum\limits_{i=1}^{n}\lambda_i-\frac{1}{2}\boldsymbol{\lambda}^T\mathbf{H}\boldsymbol{\lambda}

### Workflow
- The SVM model is initially created by specifying the type of kernel ('rbf'/'poly'/'sigmoid') and the value of the gamma parameter (by default, 'rbf' is used with gamma computed automatically during the 'fit' process).
- When the 'fit' method is called (passing a supervised training set), the model learns the correct parameters of the hyperplane by maximising a lagrangian function.
- When the 'predict' method is called, new instances are classified according to the learnt parameters.


