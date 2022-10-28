#!/usr/bin/env python
# coding: utf-8

# custom-svm
# This project aims at providing a custom implementation of Support Vector Machine, as part of the **Combinatorial
# Decision-Making and Optimization** course of the **Master in Artificial Intelligence** (*Alma Mater Studiorum*).
# This implementation is based on the **constrained non-linear optimization** of an objective function, namely the
# margin around the hyperplane which separates two clusters.
# 
# Authors:
#  - Mattia Orlandi
#  - Lorenzo Mario Amorosa

# Premises
# The implementations of the SVM and of the MulticlassSVM are contained in the `svm.py` and `multiclass_svm.py`
# modules, respectively; they provide several functionalities:
# - fitting of an SVM model for a binary classification task;
# - kernel functions in order to correctly classify non-linearly separable data;
# - soft margin in order to correctly classify semi-linearly separable data;
# - plotting of the dataset and of the separating hyperplane along multiple dimensions;
# - support to multi-class classification using a One-Versus-One approach.

# First of all, import the required libraries, in particular the custom implementations of SVM contained in the
# `svm.py` and `multiclass_svm.py` modules and the dataset generator (`sample_data_generator.py`).
# Set some useful constants, like the random state `RND`, the number of samples `N_SAMP` and the number of
# features `N_FEAT`.

# In[1]:
import matplotlib.pyplot as plt
from IPython import get_ipython
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from data.sample_data_generator import *
from multiclass_svm import MulticlassSVM
from svm import SVM

get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

RND = 42
N_SAMP = 200
N_FEAT = 2

# Moreover, define a useful function which, given a `SVM` object, the training and the test sets, 
# performs the following operations, :
# - `SVM.fit(...)` on the training set;
# - `SVM.predict(...)` on the test set;
# - computes the accuracy score;
# - plots the result (only for the custom implementation of svm).


# In[2]:
def fit_and_predict(
    svm_: SVM | SVC | MulticlassSVM,
    x_train_: np.ndarray,
    x_test_: np.ndarray,
    y_train_: np.ndarray,
    y_test_: np.ndarray,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    verbosity: int = 1,
    is_binary_custom_svm: bool = True
):
    if is_binary_custom_svm:
        svm_.fit(x_train_, y_train_, verbosity)
    else:
        svm_.fit(x_train_, y_train_)
    y_pred = svm_.predict(x_train_)
    print(f"Accuracy on the training set: {accuracy_score(y_train_, y_pred) * 100:.2f}")
    y_pred = svm_.predict(x_test_)
    print(f"Accuracy on the test set: {accuracy_score(y_test_, y_pred) * 100:.2f}")
    if is_binary_custom_svm:
        svm_.plot2d(x_train_, y_train_, x_min, x_max, y_min, y_max)

# 1. Sample datasets
# Test the SVM on a linearly separable dataset.


# In[3]:
print("=" * 50)
print("LINEARLY SEPARABLE DATASET")

x_train, x_test, y_train, y_test = linear_data_generator(
    n_samples=N_SAMP,
    n_features=N_FEAT,
    random_state=RND
)
svm = SVM(c=None)
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    x_min=-5.,
    x_max=7.,
    verbosity=2
)

print("Results displayed in 1st plot")

# As it can be seen, the SVM computes 3 support vectors, the 2 weights and the bias of the hyperplane (which is in fact
# a line in this 2D case), and predicts the label of a test set with an accuracy of 100%. In the above plot it is
# possible to see the three support vectors (circled in green), the separating line (in solid black) and the two
# margins (dotted black lines).
# Then, test the SVM on the same dataset, but fitting only on the support vectors found.


# In[4]:
print("=" * 50)
print("LINEARLY SEPARABLE DATASET - MODEL FITTED WITH SUPPORT VECTORS ONLY")

x_sv, y_sv = svm.sv_x, svm.sv_y
svm = SVM(c=None)
fit_and_predict(
    svm,
    x_sv,
    x_test,
    y_sv,
    y_test
)

print("Results displayed in 2nd plot")

# As expected the weights and the bias of the hyperplane is the same as in the previous case; the reason is that the
# support vectors are the only vectors of the dataset having strictly positive lagrangian multipliers, and thus they
# are the only vectors influencing the classifier.
# Test the SVM on a semi-linearly separable dataset which contains outliers.


# In[5]:
print("=" * 50)
print("SEMI-LINEARLY SEPARABLE DATASET - HARD MARGIN")

x_train, x_test, y_train, y_test = semi_linear_data_generator(
    n_samples=N_SAMP,
    n_features=N_FEAT,
    random_state=RND
)
svm = SVM(c=None)
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    x_min=-6.,
    x_max=6.,
    verbosity=2
)

print("Results displayed in 3rd plot")

# Due to red outliers, the separating hyperplane is very near to the blue cluster, and it has a small margin, leading
# to a model which is not robust; in fact, if new data has a high variance, some blue points could lie above the
# hyperplane and thus be wrongly classified. This is due to the use of a hard margin, not effective in the presence
# of outliers.
# Test the SVM on the same dataset but using a soft margin.


# In[6]:
print("=" * 50)
print("SEMI-LINEARLY SEPARABLE DATASET - SOFT MARGIN")

svm = SVM(c=0.1)
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    x_min=-6.,
    x_max=6.,
    verbosity=2
)

print("Results displayed in 4th plot")

# Thanks to the soft margin, the two red outliers are ignored and the separating hyperplane is placed further from
# the blue cluster than before, and with a much larger margin.
# In the next use cases, soft margin will be used (by default, $C=1$).
# Test the SVM on a non-linear separable dataset using a polynomial kernel function.


# In[7]:
print("=" * 50)
print("NON-LINEARLY SEPARABLE DATASET - POLY KERNEL")

x_train, x_test, y_train, y_test = non_linear_data_generator(
    n_samples=N_SAMP,
    random_state=RND
)
svm = SVM(kernel="poly", deg=3)
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    verbosity=1
)

print("Results displayed in 5th plot")

# First of all, since kernel functions are used it is not possible to compute the weights of the "hyperplane" (since to
# do that the mapping of $X$ in the new space must be known).
# Secondly, the polynomial kernel function does not seem adequate since the accuracy is really low; moreover, too many
# support vectors are found, and in the above plot it can be seen that the "hyperplane" does not correctly separate the
# clusters. For this dataset, a radial basis function must be used as kernel.


# In[8]:
print("=" * 50)
print("NON-LINEARLY SEPARABLE DATASET - RBF KERNEL")

svm = SVM(kernel="rbf")
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    verbosity=2
)

print("Results displayed in 6th plot")

# Thanks to the radial basis function, the accuracy is now 100% and the "hyperplane" correctly separates the two
# clusters.
# As in the linear case, test the SVM on the same dataset, but fitting only on the support vectors found.


# In[9]:
print("=" * 50)
print("NON-LINEARLY SEPARABLE DATASET - RBF KERNEL, MODEL FITTED WITH SUPPORT VECTORS ONLY")

x_sv, y_sv = svm.sv_x, svm.sv_y
svm = SVM(kernel="rbf")
fit_and_predict(
    svm,
    x_sv,
    x_test,
    y_sv,
    y_test
)

print("Results displayed in 7th plot")

# Once again fitting the model on the support vectors only does not change the result.
# 2. Sensor dataset
# The following SVM models are tested on a **dataset** constructed by simulating **measurements** of **distances**
# between **sensors** and **IoT devices** in a building.
# 
# The **building** is **composed of** a set of **rooms** on several floors, which can be automatically generated or can
# be given in input from the user. Concerning the automatic generation, it is possible to ask the generator to be more
# prone to produce a higher number of rooms per floor and vice-versa.
# In each room  it can be placed an arbitrary number of **sensors**, either manually or automatically.
# 
# The **data collection workflow** is as follows: when a **device** placed in a room **emit a signal**, the **intensity
# measured by a sensor** is inversely proportional to the square of the distance between the device and the sensor
# itself; then a Gaussian error is summed to the measured value.
# It is possible to **attach** to the **sample** either the **room** in which it is collected or its spatial position
# **coordinates**, in this way the dataset can be useful either for **classification** tasks or **regression** tasks as
# well. The implementation of the generator can be found in the python module named `sensor_data_generator.py`.
# 
# **In this notebook**, it is proposed a classification task: the SVM are trained to **predict the room** in which the
# device is located **given** the **intensities measured by sensors**.
# Utility function to show a room.


# In[10]:
def show_room(img_name: str):
    img = plt.imread(img_name)
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

# Firstly, it is addressed a **binary classification** task. It is given a floor of a building with **2 rooms** (blue
# and red) and **5 sensors**, which are represented in the following figure with numbers from 0 to 4 in whites circles.
# A dataset is sampled from this building following the steps described previously. Each sample ia labelled with the
# room where it is collected.


# In[11]:
print("=" * 50)
print("PRE-SENSE DATASET - BINARY CLASSIFICATION")
print(
    "In 8th plot the single floor with two rooms and 5 sensors is displayed; the dataset is provided by those sensors"
)

show_room("data/data_png/2_rooms.png")

# Utility function to read and process the dataset.


# In[12]:
def read_dataset(f_name: str, is_multi: bool = False):
    x_raw = []
    y_raw = []
    with open(f_name, "r") as file:
        for line in file:
            features = line.split(",")[:-1]
            x_raw.append(features)
            y_raw.append(line.split(",")[-1])
    x = np.array(x_raw).astype(float)
    y = np.array(y_raw).astype(float)
    if not is_multi:
        y = np.fromiter((-1 if y_i == 0 else 1 for y_i in y), y.dtype)
        
    return train_test_split(x, y, test_size=0.2, random_state=RND)

# Now our custom `SVM` is tested on this dataset. A radial basis kernel is chosen since the data are inherently
# non-linear.
# 
# The execution of the following cell may require some minutes since it plots the dataset and the "hyperplane" along
# every possible combination of dimensions, e.g. (0, 1), (0, 2), ..., (3,4).


# In[13]:
print(
    "WARNING: plot can take some minutes due to the fact that the 5-dimensional plot is projected "
    "onto each dimension pair (10 in total)"
)

f_name_bin = "data/data_2_rooms.txt"
x_train, x_test, y_train, y_test = read_dataset(f_name_bin)
svm = SVM(kernel="rbf")
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    x_min=0.,
    x_max=3.5,
    y_min=0.,
    y_max=7.
)

print("Results displayed in the 9th plot")

# As it can be seen, the **accuracy** is **good**, but it can still be improved, for example by displacing the sensors
# in different way or varying kernel functions. Anyway, the evaluation of tuning strategies is out of scope now.
# The **plot** can give us **further insight** on the performance: the sensors called "0" and "4" seem to be the most
# effective to discriminate between the rooms, in fact the hyperplane between them correctly separates the two
# clusters, meanwhile the sensor named "2" seems to be almost irrelevant.

# Comparison of the previous result with `sklearn.SVC`, using the same radial basis kernel (so that runtime parameters
# are the same in both cases). The evidence shows that the performance of the 2 classifiers are almost the same.


# In[14]:
print("=" * 50)
print("PRE-SENSE DATASET - BINARY CLASSIFICATION, SCIKIT-LEARN COMPARISON")

svc = SVC(kernel="rbf", gamma="scale")
fit_and_predict(
    svc,
    x_train,
    x_test,
    y_train,
    y_test,
    is_binary_custom_svm=False
)
print(f"{len(svc.support_):d} support vectors found out of {len(x_train):d} data points")

# Now it is addressed a **multiclass classification** task. It is given a building of **2 floors** composed of
# **10 rooms** in total. Inside the building **20 sensors** are placed at different height, and they are represented in
# the following figures with numbers in whites circles as in the previous case.
# A dataset is sampled from this building following the already discussed steps and each sample ia labelled with the
# room where it is collected.


# In[15]:
print("=" * 50)
print("PRE-SENSE DATASET - MULTICLASS CLASSIFICATION")
print(
    "In 10th and 11th plots the tow floor with 5 rooms and 10 sensors each are displayed; "
    "the dataset is provided by those sensors"
)

show_room("data/data_png/10_rooms_floor0.png")


# In[16]:
show_room("data/data_png/10_rooms_floor1.png")

# Here it follows the test of our `MulticlassSVM` on the dataset sampled from the previous rooms. It is chosen a radial
# basis kernel since the data are inherently non-linear, as already done in the previous case.
# 
# The execution of the following cell may require some minutes due to the size of the dataset (8000 vectors), the
# number of features (20) and the number of binary SVM needed (45 for 10 classes).


# In[17]:
print(
    "WARNING: classification may take some minutes due to the high dimensionality of the dataset (8000 samples, "
    "20 dimensions and 10 classes, for a total of 45 SVM used with a One-Vs-One approach)"
)

f_name_mul = "data/data_10_rooms.txt"
x_train, x_test, y_train, y_test = read_dataset(f_name_mul, is_multi=True)
svm = MulticlassSVM(kernel="rbf")
fit_and_predict(
    svm,
    x_train,
    x_test,
    y_train,
    y_test,
    is_binary_custom_svm=False
)

# Comparison of the previous result with `sklearn.SVC` for multiclass classification. It is used again a radial basis
# kernel and the strategy one-versus-one for multiclass classification is specified (so that runtime parameters are the
# same in both cases). The evidence shows that the performance of the 2 classifiers are almost the same in this
# case as well.


# In[18]:
print("=" * 50)
print("PRE-SENSE DATASET - MULTICLASS CLASSIFICATION, SCIKIT-LEARN COMPARISON")

svc = SVC(kernel="rbf", gamma="scale", decision_function_shape="ovo")
fit_and_predict(
    svc,
    x_train,
    x_test,
    y_train,
    y_test,
    is_binary_custom_svm=False
)
print(f"{len(svc.support_):d} support vectors found out of {len(x_train):d} data points")
