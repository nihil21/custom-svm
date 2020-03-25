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
 - the module [`custom-svm/svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py) contains the implementation of SVM for binary classification, with supports to kernel functions and soft margin.  
 - the module [`custom-svm/multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/multiclass_svm.py) contains the implementation of SVM for multiclass classification.
 - the notebook [`custom-svm/svm_usecase.ipynb`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm_usecase.ipynb) shows the usage of the SVM for many different purposes.
 - the package [`custom-svm/data`](https://github.com/nihil21/custom-svm/tree/master/custom-svm/data) contains generators and datasets. 

### Lagrangian Formulation of the SVM

The Lagrangian problem for SVM formulated in its dual form:

<img src="https://latex.codecogs.com/gif.latex?max%5C%2C%20F%28%5Cmathbf%7B%5Clambda%7D%29%20%3D%20%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%5Calpha_i-%5Cfrac%7B1%7D%7B2%7D%5Csum%5Climits_%7Bi%3D1%7D%5E%7Bn%7D%5Csum%5Climits_%7Bj%3D1%7D%5E%7Bn%7D%5Calpha_i%5Calpha_j%5C%2C%20y_i%5C%2C%20y_j%5Cmathbf%7B%5C%2C%20x_i%5C%2C%20x_j%7D">    

It is a quadratic optimization problem that can be solved using the quadratic library `cvxopt`. In details, the signature of the solver .... to be continued 


-----------------------------------------------
link latex generator: https://www.codecogs.com/latex/eqneditor.php

### Workflow
- The SVM model is initially created by specifying the type of kernel ('rbf'/'poly'/'sigmoid') and the value of the gamma parameter (by default, 'rbf' is used with gamma computed automatically during the 'fit' process).
- When the 'fit' method is called (passing a supervised training set), the model learns the correct parameters of the hyperplane by maximising a lagrangian function.
- When the 'predict' method is called, new instances are classified according to the learnt parameters.


