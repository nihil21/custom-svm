# custom-svm
Custom implementation of Support Vector Machines using Python and NumPy, as part of the Combinatorial Decision Making and Optimization university course (Master in Artificial Intelligence, Alma Mater Studiorum - University of Bologna).

### Authors
[Mattia Orlandi](https://github.com/nihil21)  
[Lorenzo Mario Amorosa](https://github.com/Lostefra)

### Requirements
This project requires the following libraries:
- `numpy` for basic operations on matrices;
- `cvxopt` for the quadratic convex optimization;
- `scikit-learn` for generating and splitting the dataset, to assess accuracy, and to confront our implementation with SVC;
- `matplotlib` for plotting graphs.

The complete list of the packages used in the virtual environment is in [`requirements.txt`](https://github.com/nihil21/custom-svm/blob/master/requirements.txt); to install all those modules, it is sufficient to issue the command `pip install -r requirements.txt` (better if done in an Anaconda environment).

### Design and Implementation: Overview

The repository is structured in the following way:
 - the module [`custom-svm/svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py) contains the implementation of SVM for binary classification, with support to kernel functions and soft margin.  
 - the module [`custom-svm/multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/multiclass_svm.py) contains the implementation of SVM for multiclass classification.
 - the notebook [`custom-svm/svm_usecase.ipynb`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm_usecase.ipynb) shows the usage of the SVM for many different purposes.
 - the package [`custom-svm/data`](https://github.com/nihil21/custom-svm/tree/master/custom-svm/data) contains generators and datasets.

We provided also a script version ([`custom-svm/svm_usecase.py`](https://github.com/nihil21/custom-svm/tree/master/custom-svm/svm_usecase.py)) of the Jupyter notebook which can be run either in a terminal or in Spyder (recommended). For more clarity, it is suggested to at least read the notebook comments.

### Lagrangian Formulation of the SVM and Optimization

The Lagrangian problem for SVM is formulated as follows:

![Latex image not found :(](res/lag_p.gif?raw=true)

To integrate the soft margin in the formulation, for each data point ![](res/x_i_inline.gif?raw=true) a variable ![](res/xi_i.gif?raw=true) is introduced; such variable represents the distance of ![](res/x_i_inline.gif?raw=true) from the corresponding class margin if ![](res/x_i_inline.gif?raw=true) lies on the wrong side of such margin, otherwise they are zero. In other words, ![](res/xi_i.gif?raw=true) represents the penalty of the misclassified data point ![](res/x_i_inline.gif?raw=true), and ![](res/C.gif?raw=true) controls the trade-off between the amount of misclassified samples and the size of the margin.

Every point ![](res/x_i_inline.gif?raw=true) must satisfy the following constraint:

![Latex image not found :(](res/xi_const.gif?raw=true)

By integrating it into the Lagrangian, the following is obtained:

![Latex image not found :(](res/lag_p_soft.gif?raw=true)

Its dual problem is formulated as follows:

![LaTeX image not found :(](res/dual.gif?raw=true)

subject to:   

![LaTeX image not found :(](res/const1_C.gif?raw=true)

![LaTeX image not found :(](res/const2.gif?raw=true)    

It is a quadratic optimization problem that can be solved using the quadratic library `cvxopt` in python, so it is necessary to match the solver's API which, according to the documentation, is of the form:

![LaTeX image not found :(](res/cvxopt_sign.gif?raw=true)

subject to:  

![LaTeX image not found :(](res/const3.gif?raw=true)   

![LaTeX image not found :(](res/const4.gif?raw=true)

Let ![](res/H.gif?raw=true) be a matrix such that ![LaTeX image not found :(](res/inline_h.gif?raw=true) , then the function to optimize becomes:

![LaTeX image not found :(](res/dual_h.gif?raw=true)

We then convert the sums into vector form and multiply both the objective and the constraint by −1, which turns this into a minimization problem and reverses the inequality in constaints. The optimization problem can be written as:

![LaTeX image not found :(](res/dual_h2.gif?raw=true)

subject to:  

![LaTeX image not found :(](res/const5.gif?raw=true)   

![LaTeX image not found :(](res/const_7.gif?raw=true)   

![LaTeX image not found :(](res/const6.gif?raw=true)    

It is now necessary to convert the numpy arrays that express the optimization problem accordingly to `cvxopt` format. Supposed m the number of samples and using the same notation as in the documentation, this gives:  
 - ![LaTeX image not found :(](res/inline1.gif?raw=true) a matrix of size m×m
 - ![LaTeX image not found :(](res/inline2.gif?raw=true)  a vector of size m×1
 - ![LaTeX image not found :(](res/inline3.gif?raw=true) a matrix of size 2m×m, such that a diagonal matrix of -1s of size m×m is concatenated vertically with another diagonal matrix of 1s of size m×m
 - ![LaTeX image not found :(](res/inline4.gif?raw=true)  a vector of size 2m×1, with zeros in the first m cells and C in the other m cells
 - ![LaTeX image not found :(](res/inline5.gif?raw=true) the label vector of size m×1
 - ![LaTeX image not found :(](res/inline6.gif?raw=true) a scalar  

It has to be noticed that in case of hard margin the constraints on the upper bound of the Lagrangian multipliers are not given, hence ![](res/G.gif?raw=true) and ![](res/small_h.gif?raw=true) are smaller in that case.  

#### Kernel trick
Since the hyperplane is a linear function, the SVM model defined so far is suited only to linearly separable datasets, which is not very useful in real-world scenarios.  
To enable the correct classification in the non-linear case, the data to classify is mapped by ![](res/map_x.gif?raw=true) into a new space, in which the data is linearly separable and thus in which SVM can be applied.

However, computing the mapping ![](res/map_x.gif?raw=true) for every ![](res/x.gif?raw=true) is computationally expensive; therefore, since only the product ![](res/dot_prod.gif?raw=true) is relevant as far as fitting and classification are concerned, only the mapping of such product is considered (*kernel trick*):

![](res/kernel_trick.gif?raw=true)

where ![](res/K.gif?raw=true) is called *kernel function*, and it can be:
- dot product (**linear case**);
- polynomial;
- radial basis function;
- sigmoid.

In the [`python code`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py) the parameters needed by the solver are defined as follows, using the guideline previously provided:
 ```python
        K = np.zeros(shape=(n_samples, n_samples))
        for i, j in itertools.product(range(n_samples), range(n_samples)):
            K[i, j] = self.kernel_fn(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        # Compute G and h matrix according to the type of margin used
        if self.C:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples),
                                         np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples),
                                         np.ones(n_samples) * self.C)))
        else:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        A = cvxopt.matrix(y.reshape(1, -1).astype(np.double))
        b = cvxopt.matrix(np.zeros(1))

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
```
 The support vectors can be get exploiting the variable `sol`, which are those with positive Lagrangian multipliers.

 ```python
        lambdas = np.ravel(sol['x'])
        is_sv = lambdas > 1e-5
        self.sv_X = X[is_sv]
        self.sv_y = y[is_sv]
        self.lambdas = lambdas[is_sv]
 ```

 ### Computation of the separating hyperplane   

 It is possible to compute then ![](res/w.gif?raw=true), if the kernel is linear, and ![](res/b.gif?raw=true), which are the parameters of the "hyperplane" which separates the classes, in fact:

 ![LaTeX image not found :(](res/w_hyp.gif?raw=true)  

 And given ![](res/S.gif?raw=true) as the set of the support vectors:   

 ![LaTeX image not found :(](res/b_hyp.gif?raw=true)   

 In the [`python code`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py) the computation is made as follows:
 ```python
         self.w = np.zeros(n_features)
         for i in range(len(self.lambdas)):
             self.w += self.lambdas[i] * self.sv_X[i] * self.sv_y[i]
```
```python         
        self.b = 0
        for i in range(len(self.lambdas)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lambdas * self.sv_y * K[sv_index[i], is_sv])
        self.b /= len(self.lambdas)
```


 ### Prediction of the class label

 Supposed S the number of support vectors, an input ![](res/x.gif?raw=true) is assignment to a class label ![](res/y.gif?raw=true) with the following formula. As a side node, in case of linear kernel taking simply the dot product between input and support vectors is enough.     

 ![LaTeX image not found :(](res/pred.gif?raw=true)  

 In [`code`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py):   

 ```python
         y_predict = 0
         for lamda, sv_X, sv_y in zip(self.lambdas, self.sv_X, self.sv_y):
                 y_predict += lamda * sv_y * self.kernel_fn(X, sv_X)
         y_predict = np.sign(y_predict + self.b)
```

### SVM for Multiclass Classification

The module [`multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/multiclass_svm.py) contains the implementation of Support Vector Machine for multi-classification purposes based on **one-vs-one strategy**.  
It offers full support to **kernel functions** and **soft margin**, in fact the signature of its `__init__` method is the same of the binary `SVM`.    
Given N different classes to classify, the algorithm provides ![LaTeX image not found :(](res/multi.gif?raw=true) SVM binary classifiers from the module [`svm.py`](https://github.com/nihil21/custom-svm/blob/master/custom-svm/svm.py).   
**Each classifier** is **trained** to correctly classify **2 of the N** given **classes**. In the training process there are used only the entries in the dataset to which it corresponds a label of the 2 classes.   
Given an unseen example, the **prediction** of the class is computed deploying a **voting schema** among the binary `SVM` classifiers.   
The voting process is based on the standard `predict` function for binary `SVM` classifiers, so the tested entry is assigned to the class which wins the highest number of binary comparisons. In addition, it is available a mechanism to **counteract** the possible risk of **draw** in voting, based on the raw values predicted by the binary classifiers before the application of 'sign' function.

### Workflow
- The SVM model is initially created by specifying the type of kernel ('rbf'/'poly'/'sigmoid') and the value of the associated parameters ('gamma', 'deg' and 'r'); also, the parameter 'C' regulating the soft margin is specified.
- When the 'fit' method is called (passing a supervised training set), the model learns the correct parameters of the hyperplane by minimizing the dual lagrangian function discussed in the previous section.
- When the 'predict' method is called, new instances are classified according to the learnt parameters.

### Credits
[Tristan Fletcher, Support Vector Machines Explained](https://static1.squarespace.com/static/58851af9ebbd1a30e98fb283/t/58902fbae4fcb5398aeb7505/1485844411772/SVM+Explained.pdf)     
[Humboldt-Universität zu Berlin, Lagrangian formulation of the SVM](http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/stfhtmlnode64.html)
