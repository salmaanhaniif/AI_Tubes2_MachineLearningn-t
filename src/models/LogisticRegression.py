import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=100, threshold=0.5):
        # inisiasi komponen
        self.learning_rate = learning_rate
        self.n_iteration = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def fit(self, x, y):
        # malatih model dengan stochastic gradient descent
        n_samples = x.shape[0]

        # bias diinisiasi 1
        x_biased = np.hstack((np.ones((n_samples, 1)), x))

        n_total_features = x_biased.shape[1]

        self.weights = np.zeros(n_total_features)
        for _ in range (self.n_iteration):
            # urutan data  dirandom 
            shuffled_indice = np.random.permutation(n_samples)
            x_shuffeled = x_biased[shuffled_indice]
            y_shuffeled = y[shuffled_indice]

            for i in range(n_samples):
                xi = x_shuffeled[i]
                yi = y_shuffeled[i]

                # menghitung z
                z = np.dot(xi, self.weights)

                # menghitung probabilitas 
                pi = self.sigmoid(z)
                
                # update bobotnya
                gradient = (yi - pi) * xi
                self.weights += self.learning_rate * gradient

    # untuk prediksi data test
    def predict_probability(self, x):
        n_samples = x.shape[0]
        x_biased = np.hstack((np.ones((n_samples, 1)), x))

        linear_model = x_biased.dot(self.weights)
        probabilities = self.sigmoid(linear_model)
        return probabilities
    
    def predict(self, x):
        probabilities = self.predict_probability(x)
        predictions = np.where(probabilities >= self.threshold, 1, 0)
        return predictions
