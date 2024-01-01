# PCA Algorithm Project

## Introduction

This project implements a Principal Component Analysis (PCA) algorithm for dimensionality reduction. The PCA algorithm is implemented in the dim_reduction.py module, and it provides methods for fitting the model, transforming data, and inverse transforming data.


## Mathematics Behind PCA Algorithm

### 1. Covariance Matrix

The first step in PCA involves computing the covariance matrix of the centered data. Given a data matrix X with dimensions (m, n) where m is the number of samples and n is the number of features, the covariance matrix C is computed as follows:

C = (1 / (m - 1)) * (X - X̄)^T * (X - X̄)

where X̄ is the mean vector of the features.

### 2. Eigenvectors and Eigenvalues

After obtaining the covariance matrix, the next step is to compute the eigenvectors and eigenvalues. The eigenvectors V and eigenvalues D can be obtained by solving the following equation:

C * V = D * V

The eigenvectors represent the directions of maximum variance, and the corresponding eigenvalues indicate the magnitude of variance along those directions.

### 3. Sorting and Selecting Components

The eigenvectors are sorted based on their corresponding eigenvalues in descending order. The sorted eigenvectors represent the principal components of the data. The user can choose to retain a certain number of principal components or specify the variance to be retained.

## Pseudocode of PCA Algorithm

```
Initialize PCA:
  - Set the desired number of components (default to 2)
  - Set mean and components to None

Fit the model to input data:
  - Center the data by subtracting the mean of each feature
  - Calculate the covariance matrix of the centered data
  - Obtain eigenvectors and eigenvalues from the covariance matrix
  - Sort eigenvectors in descending order based on eigenvalues
  - Select the desired number of components based on user input or desired variance

Transform data to the reduced feature space:
  - Center the data using the mean
  - Project the centered data onto the selected principal components

Inverse transform the reduced data:
  - Project the reduced data back to the original feature space

End
```


## Module 1: dim_reduction.py

### 1. PCA Class

The PCA class is designed to perform Principal Component Analysis on input data.

### 2. Methods

* __init__(self, n_components=2): Constructor method to initialize the PCA object.
* fit(self, X): Fit the PCA model to the input data.
* transform(self, X): Transform the input data into the reduced feature space.
* fit_transform(self, X): Fit the model and transform the input data simultaneously.
* inverse_transform(self, X): Inverse transform the reduced data back to the original feature space.

### 3. Example Usage

```
from dim_reduction import PCA
import numpy as np

# Create PCA object with 2 components
pca = PCA(n_components=2)

# Fit and transform data
X_transformed = pca.fit_transform(X)

# Inverse transform data
X_original = pca.inverse_transform(X_transformed)
```

## Module 2: datasets.py

This module provides functions to generate synthetic datasets for testing the PCA algorithm.

### 1. Functions

* random_init(size_in, size_out): Returns a random array of the given shape.
* print_shape(X, y): Prints the shape of the feature and target arrays.
* make_classification(...): Generates a random n-class classification problem.
* make_blobs(...): Generates a blob dataset with specified parameters.
* make_circles(...): Generates a dataset with circular clusters.

### 2. Example Usage

```
from datasets import make_classification, make_blobs, make_circles
import numpy as np

# Generate a classification dataset
X, y = make_classification(n_samples=100, n_features=20, n_classes=2)

# Generate a blob dataset
X_blob, y_blob = make_blobs(n_samples=100, n_features=20, n_centers=2)

# Generate a dataset with circular clusters
X_circles, y_circles = make_circles(n_samples=100, n_features=2)
```

## Module 3: plots.py

This module provides functions to visualize the generated datasets.

### 1. Functions

* scatter2D(title, xlabel, ylabel, X, y): Creates a 2D scatter plot of the input data.
* scatter3D(title, xlabel, ylabel, zlabel, X, y): Creates a 3D scatter plot of the input data.

### 2. Example Usage

```
from plots import scatter2D, scatter3D
import numpy as np

# Visualize 2D scatter plot
scatter2D("2D Scatter Plot", "X-axis", "Y-axis", X, y)

# Visualize 3D scatter plot
scatter3D("3D Scatter Plot", "X-axis", "Y-axis", "Z-axis", X, y)
```
## Notebook Example

For a complete example of using the PCA algorithm on synthetic datasets and visualizing the results, refer to the provided notebook example_notebook.ipynb.

## Usage
To use the project, Make sure to install necessary dependencies by running pip install numpy matplotlib before executing the code in the notebook.

## Contributing
The project welcomes contributions from other users. They can open an issue or submit a pull request with their ideas or changes.

## License
The project is licensed under the terms of the MIT license.





