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

### Design and Implementation: Overview

The repository is structured in the following way:
 - the module [`src/svm.py`](https://github.com/nihil21/custom-svm/blob/master/src/svm.py) contains the implementation of SVM for binary classification, with support to kernel functions and soft margin.  
 - the module [`src/multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/src/multiclass_svm.py) contains the implementation of SVM for multiclass classification.
 - the notebook [`src/svm_usecase.ipynb`](https://github.com/nihil21/custom-svm/blob/master/src/svm_usecase.ipynb) shows the usage of the SVM for many different purposes.
 - the package [`src/data`](https://github.com/nihil21/custom-svm/tree/master/src/data) contains generators and datasets.

We provided also a script version ([`src/svm_usecase.py`](https://github.com/nihil21/custom-svm/tree/master/src/svm_usecase.py)) of the Jupyter notebook which can be run either in a terminal or in Spyder (recommended). For more clarity, it is suggested to at least read the notebook comments.

### Lagrangian Formulation of the SVM and Optimization

The Lagrangian problem for SVM is formulated as follows:

$$ \min L(\mathbf{w}, b, \mathbf{\Lambda}) = \frac{1}{2} \|\mathbf{w}\|^2 + \sum_{i=1}^n \lambda_i(y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1) $$

To integrate the soft margin in the formulation, for each data point $\mathbf{x}_i$ a variable $\xi_i$ is introduced; such variable represents the distance of $\mathbf{x}_i$ from the corresponding class margin if $\mathbf{x}_i$ lies on the wrong side of such margin, otherwise they are zero. In other words, $\xi_i$ is represents the penalty of the misclassified data point $\mathbf{x}_i$, and $C$ controls the trade-off between the amount of misclassified samples and the size of the margin.

Every point $\mathbf{x}_i$ must satisfy the following constraint:

$$ y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 - \xi_i $$

By integrating it into the Lagrangian, the following is obtained:

$$ \min L(\mathbf{w}, b, \mathbf{\Lambda}) = \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i + \sum_{i=1}^n \lambda_i(y_i(\mathbf{w} \cdot \mathbf{x}_i + b) - 1 + \xi_i) $$

Its dual problem is formulated as follows:

$$ \max F(\mathbf{\Lambda}) = \sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \lambda_i \lambda_j y_i y_j \mathbf{x}_i\cdot\mathbf{x}_j $$

subject to:   

$$
\begin{align}
    &0 \le \lambda_i \le C,\;\;\;\;i = 1, ..., n \\
    &\sum_{i=1}^n \lambda_i y_i = 0
\end{align}
$$

It is a quadratic optimization problem that can be solved using the quadratic library `cvxopt` in python, so it is necessary to match the solver's API which, according to the documentation, is of the form:

$$ \min_\mathbf{x} F(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\intercal\mathbf{Px} + \mathbf{q}^\intercal\mathbf{x} $$

subject to:  

$$
\begin{align}
    &\mathbf{Ax} = \mathbf{b} \\
    &\mathbf{Gx} \le \mathbf{h}
\end{align}
$$

Let $\mathbf{H}$ be a matrix such that $\mathbf{H}_{i,j}=y_i y_j \mathbf{x}_i\cdot\mathbf{x}_j$, then the function to optimize becomes:

$$ \max_\mathbf{\Lambda} F(\mathbf{\Lambda}) = \sum_{i=1}^n \lambda_i - \frac{1}{2}\mathbf{\Lambda}^\intercal\mathbf{H\Lambda} $$

We then convert the sums into vector form and multiply both the objective and the constraint by −1, which turns this into a minimization problem and reverses the inequality in constaints. The optimization problem can be written as:

$$ \min_\mathbf{\Lambda} F(\mathbf{\Lambda}) = \frac{1}{2}\mathbf{\Lambda}^\intercal\mathbf{H\Lambda} - \mathbf{1}^\intercal\mathbf{\Lambda} $$

subject to:

$$
\begin{align}
    &-\lambda_i \le 0 \\
    &\lambda_i \le C \\
    &\mathbf{y}^\intercal\mathbf{\Lambda} = 0
\end{align}
$$

It is now necessary to convert the numpy arrays that express the optimization problem accordingly to `cvxopt` format. Supposed $m$ the number of samples and using the same notation as in the documentation, this gives:  
 - $\mathbf{P}:=\mathbf{H}$ a matrix of size $m \times m$;
 - $\mathbf{q}:=-\mathbf{1}$ a vector of size $m \times 1$;
 - $\mathbf{G}:=$ a matrix of size $2m \times m$, such that a diagonal matrix of $-1$s of size $m \times m$ is concatenated vertically with another diagonal matrix of $1$s of size $m \times m$;
 - $\mathbf{h}:=\mathbf{0}$ a vector of size $2m \times 1$, with $0$s in the first $m$ cells and $C$s in the other $m$ cells;
 - $\mathbf{A}:=\mathbf{y}$ the label vector of size $m \times 1$;
 - $b:=0$ a scalar.

It has to be noticed that in case of hard margin the constraints on the upper bound of the Lagrangian multipliers are not given, hence $\mathbf{G}$ and $\mathbf{h}$ are smaller in that case.  

#### Kernel trick
Since the hyperplane is a linear function, the SVM model defined so far is suited only to linearly separable datasets, which is not very useful in real-world scenarios.  
To enable the correct classification in the non-linear case, the data to classify is mapped by $\phi(\mathbf{x})$ into a new space, in which the data is linearly separable and thus in which SVM can be applied.

However, computing the mapping $\phi(\mathbf{x})$ for every $\mathbf{x}$ is computationally expensive; therefore, since only the product $\mathbf{x}_i\cdot\mathbf{x}_j$ is relevant as far as fitting and classification are concerned, only the mapping of such product is considered (*kernel trick*):

$$ K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)\cdot\phi(\mathbf{x}_j) $$

where $K$ is called *kernel function*, and it can be:
- dot product (**linear case**);
- polynomial;
- radial basis function;
- sigmoid.

In the [`python code`](https://github.com/nihil21/custom-svm/blob/master/src/svm.py) the parameters needed by the solver are defined as follows, using the guideline previously provided:
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

Then, it is possible to compute $\mathbf{w}$, if the kernel is linear, and $b$, which are the parameters of the "hyperplane" which separates the classes; in fact:

$$ \mathbf{w}=\sum_{i=1}^n \lambda_i y_i \mathbf{x}_i $$

And given $S$ as the set of the support vectors:   

$$ b = \frac{1}{N_S}\sum_{s \in S} \left(y_s - \sum_{m \in S} \lambda_m y_m K(\mathbf{x}_m, \mathbf{x}_s) \right) $$

In the [`python code`](https://github.com/nihil21/custom-svm/blob/master/src/svm.py) the computation is made as follows:
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

An input $\mathbf{x}$ is assignment to a class label $y$ with the following formula. As a side node, in case of linear kernel taking simply the dot product between input and support vectors is enough.     

$$ y = \text{sgn} \left( \sum_{i=1}^n \lambda_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right) $$

In [`code`](https://github.com/nihil21/custom-svm/blob/master/src/svm.py):   

```python
        y_predict = 0
        for lamda, sv_X, sv_y in zip(self.lambdas, self.sv_X, self.sv_y):
                y_predict += lamda * sv_y * self.kernel_fn(X, sv_X)
        y_predict = np.sign(y_predict + self.b)
```

### SVM for Multiclass Classification

The module [`multiclass_svm.py`](https://github.com/nihil21/custom-svm/blob/master/src/multiclass_svm.py) contains the implementation of Support Vector Machine for multi-classification purposes based on **one-vs-one strategy**.  
It offers full support to **kernel functions** and **soft margin**, in fact the signature of its `__init__` method is the same of the binary `SVM`.    
Given $N$ different classes to classify, the algorithm provides $N(N-1)/2$ SVM binary classifiers from the module [`svm.py`](https://github.com/nihil21/custom-svm/blob/master/src/svm.py).   
**Each classifier** is **trained** to correctly classify **2 of the N** given **classes**. In the training process there are used only the entries in the dataset to which it corresponds a label of the 2 classes.   
Given an unseen example, the **prediction** of the class is computed deploying a **voting schema** among the binary `SVM` classifiers.   
The voting process is based on the standard `predict` function for binary `SVM` classifiers, so the tested entry is assigned to the class which wins the highest number of binary comparisons. In addition, it is available a mechanism to **counteract** the possible risk of **draw** in voting, based on the raw values predicted by the binary classifiers before the application of 'sign' function.

### Workflow
- The SVM model is initially created by specifying the type of kernel (`rbf`/`poly`/`sigmoid`) and the value of the associated parameters (`gamma`, `deg` and `r`); also, the parameter `C` regulating the soft margin is specified.
- When the `fit` method is called (passing a supervised training set), the model learns the correct parameters of the hyperplane by minimizing the dual lagrangian function discussed in the previous section.
- When the `predict` method is called, new instances are classified according to the learnt parameters.

### Credits
[Tristan Fletcher, Support Vector Machines Explained](https://static1.squarespace.com/static/58851af9ebbd1a30e98fb283/t/58902fbae4fcb5398aeb7505/1485844411772/SVM+Explained.pdf)     
[Humboldt-Universität zu Berlin, Lagrangian formulation of the SVM](http://sfb649.wiwi.hu-berlin.de/fedc_homepage/xplore/tutorials/stfhtmlnode64.html)
