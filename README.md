# custom-svm
Custom implementation of Support Vector Machines using Python and NumPy, as part of the Combinatorial Decision Making and Optimization university course (Master in Artificial Intelligence, Alma Mater Studiorum - University of Bologna).

### Workflow
- The SVM model is initially created by specifying the type of kernel ('rbf'/'poly'/'sigmoid') and the value of the gamma parameter (by default, 'rbf' is used with gamma computed automatically during the 'fit' process).
- When the 'fit' method is called (passing a supervised training set), the model learns the correct parameters of the hyperplane by maximising a lagrangian function.
- When the 'predict' method is called, new instances are classified according to the learnt parameters.

### Credits
Mattia Orlandi
Lorenzo Mario Amorosa
