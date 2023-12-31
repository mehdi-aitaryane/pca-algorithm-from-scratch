import numpy as np

# This function returns a random array of the given shape
# size_in: the number of rows of the array
# size_out: the number of columns of the array

def random_init(size_in, size_out):
    return np.random.uniform(0.0, 1.0, size=(size_in, size_out))

# This function prints the shape of the feature and target arrays
# X: a 2D array of features
# y: a 1D array of targets

def print_shape(X, y):
    print("X shape is ", X.shape)
    print("y shape is ", y.shape)


# This function generates a random n-class classification problem.
# Parameters:
# n_samples: int, Number of samples
# n_features: int, Number of features
# n_classes: int, Number of classes
# noise: float, Standard deviation of Gaussian noise added to the features
# balanced: bool, Whether to generate balanced samples
# random_state: int, Seed for the random number generator
# init: function, Function to initialize weights and bias
# Returns: tuple, A tuple containing the feature matrix X and the target vector y

def make_classification(n_samples=101, n_features=20, n_classes=2, noise=0.0, balanced=True, random_state=None, init = random_init):
    # Select random state
    np.random.seed(random_state)

    # Create balanaced or unbalanced samples
    size = n_samples // n_classes
    rest = n_samples % n_classes
    prob = np.random.uniform(size=(n_classes))
    prob = np.where(balanced == True, [1/n_classes] * n_classes, prob / sum(prob))

    # Generate target values
    y = np.random.choice(n_classes, n_samples, replace=True, p=prob)

    # Initialize weights and bias
    weights = init(1, n_features)[0]
    bias = init(1, 1)[0]

    # Generate random features
    X = np.random.uniform(size=(n_samples, n_features))

    # Add noise to features before any adjustments
    X += np.random.normal(scale=noise, size=(n_samples, n_features))

    # Calculate y_pred such that X * weights + bias = y
    y_pred = np.dot(X, weights) + bias

    # Adjust X to satisfy the equation
    X += (y - y_pred)[:, np.newaxis] * weights
    return X, y

# This function generates a blob dataset with a specified number of samples, features, centers, radii, balance, noise, and random state
# n_samples: the number of samples in the dataset
# n_features: the number of features in the dataset
# n_centers: the number of clusters or centers in the dataset
# low_radius: the lower bound of the radii of the clusters
# high_radius: the upper bound of the radii of the clusters
# balanced: a boolean value indicating whether the clusters have equal probabilities or not
# noise: the standard deviation of the noise added to the features
# random_state: the seed for the random number generator

def make_blobs(n_samples=100, n_features=20, n_centers=2, low_radius = 0.01, high_radius = 0.035, balanced = True, noise = 0.0, random_state=None):
    np.random.seed(random_state)

    # Create balanaced or unbalanced samples
    size = n_samples // n_centers
    rest = n_samples % n_centers
    prob = np.random.uniform(size=(n_centers))
    prob = np.where(balanced == True, [1/n_centers] * n_centers, prob / sum(prob))
    # Generate target values
    y = np.random.choice(n_centers, n_samples, replace=True, p=prob)
    y = np.sort(y)
    blob_centers = np.random.uniform(size=(n_centers, n_features))
    blob_radius = np.random.uniform(low=low_radius, high=high_radius, size=n_centers)
    class_sizes = np.bincount(y)

    features = [center + radius * np.random.randn(samples, n_features)
         for center, radius, samples in zip(blob_centers, blob_radius, class_sizes)]
    
    X = np.concatenate(features)
    X += np.random.normal(scale=noise, size=(n_samples, n_features))
    return X, y

def make_circles(n_samples=100, noise=0.0, random_state=None, n_circles=2, balanced = True, n_features=2):
    np.random.seed(random_state)
    # Create balanced or unbalanced samples
    size = n_samples // n_circles
    rest = n_samples % n_circles
    prob = np.random.uniform(size=(n_circles))
    prob = np.where(balanced == True, [1/n_circles] * n_circles, prob / sum(prob))
    # Generate target values
    Y = np.random.choice(n_circles, n_samples , replace=True, p=prob)
    Y = np.sort(Y)
    class_sizes = np.bincount(Y)
    factors = np.sort(np.random.uniform(size=(n_circles,)))
    X = []
    for i in range(n_circles):
        theta = np.random.rand(class_sizes[i]) * 2 * np.pi
        r = np.random.uniform(low=factors[i], high=factors[i], size=(class_sizes[i]))
        x, y = r * np.cos(theta), r * np.sin(theta)
        X.append(np.vstack((x, y)).T)
    X = np.vstack(X)
    if n_features > 2:
        extra_features = np.random.uniform(size=(n_samples, n_features - 2))
        X = np.hstack((X, extra_features))
    noise = np.random.normal(scale=noise, size=(n_samples, n_features))
    X = X + noise
    return X, Y