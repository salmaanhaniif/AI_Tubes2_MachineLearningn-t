import numpy as np
from numba import jit

"""

Rasionalisasi gue pake numba:
- in this type of algo, using numba jit just make sense, unless pake mini batch yg udh full vectorized (which im considering to make dan opsional)
- gue nyoba pake dataset titanic yg cleaned which i found online, sebenernya dari segi akurasi, yg ori tuh udah kelasss abis bahkan sama
  kayak yg punya scikit. Tapi, dari segi waktu, yang punya scikit cuma 0-0.1 sec meanwhile yg punya kita 5.5-6.5 sec. After pake numba jit + vector mult 
  dr numpy (yg @) (jujur gk tau yg ini sengaruh apa) jadi drop jadi 0.5-1.5 sec doang tanpa ada perubahan dari segi akurasi.
  
"""

@jit(nopython=True)
def sgd_update_numba(x_shuffled, y_shuffled, weights, learning_rate, n_samples):
    for i in range(n_samples):
        xi = x_shuffled[i]
        yi = float(y_shuffled[i])  
        
        # Itung z dan probabilitas
        z = 0.0
        for j in range(len(xi)):
            z += xi[j] * weights[j]
        
        # Sigmoid
        z_clipped = max(min(z, 500.0), -500.0)
        pi = 1.0 / (1.0 + np.exp(-z_clipped))
        
        # Update weights (gradient ascent)
        error = yi - pi
        for j in range(len(xi)):
            weights[j] += learning_rate * error * xi[j]
    
    return weights



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

            self.weights = sgd_update_numba(
                x_shuffeled, y_shuffeled, self.weights, 
                self.learning_rate, n_samples
            )

    # untuk prediksi data test
    def predict_probability(self, x):
        n_samples = x.shape[0]
        x_biased = np.hstack((np.ones((n_samples, 1)), x))

        linear_model = x_biased @ self.weights 
        probabilities = self.sigmoid(linear_model)
        return probabilities
    
    def predict(self, x):
        probabilities = self.predict_probability(x)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions
