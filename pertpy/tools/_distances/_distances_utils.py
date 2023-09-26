import numpy as np

class DistanceBootstrapper:
    def __init__(self, distance_metric, n_bootstraps=1000):
        self.distance_metric = distance_metric
        self.n_bootstraps = n_bootstraps

    def calculate_variance(self, X, Y, **kwargs):
        distances = []
        for _ in range(self.n_bootstraps):
            # Generate bootstrapped samples
            X_bootstrapped = np.random.choice(X, len(X), replace=True)
            Y_bootstrapped = np.random.choice(Y, len(Y), replace=True)

            # Calculate the distance using the provided distance metric
            distance = self.distance_metric(X_bootstrapped, Y_bootstrapped, **kwargs)
            distances.append(distance)

        # Calculate the variance of the distances
        variance = np.var(distances)
        return variance


class ThreeWayComparison:
    """First draft for a ThreeWayComparison class.
    TODOs:
    - evaluate if something like this is doing what we want
    - if yes, integrate into pairwise comparison (and one-sided)
    """
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def compare_X_to_Y(self, X, Y, **kwargs):
        # Calculate the distance using the provided distance metric
        distance = self.distance_metric(X, Y, **kwargs)
        return distance

    def compare_X_to_Z(self, X, Z, **kwargs):
        # Calculate the distance using the provided distance metric
        distance = self.distance_metric(X, Z, **kwargs)
        return distance
    
    def __call__(self, X, Y, Z):
        return self.compare_X_to_Y(X, Y), self.compare_X_to_Z(X, Z)