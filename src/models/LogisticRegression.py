import numpy as np
from numba import jit

"""

Rasionalisasi gue pake numba:
- in this type of algo, using numba jit just make sense, unless pake mini batch yg udh full vectorized (which im considering to make dan opsional)
- gue nyoba pake dataset titanic yg cleaned which i found online, sebenernya dari segi akurasi, yg ori tuh udah kelasss abis bahkan sama
  kayak yg punya scikit. Tapi, dari segi waktu, yang punya scikit cuma 0-0.1 sec meanwhile yg punya kita 5.5-6.5 sec. After pake numba jit + vector mult 
  dr numpy (yg @) (jujur gk tau yg ini sengaruh apa) jadi drop jadi 1-1.5 sec doang tanpa ada perubahan dari segi akurasi.

Findings after coba optimasi pake mini batch instead of stochastic (pake dataset yg titanic) :
- waktu jadi drop antara 0.5-1 which yaaa beda 0.5 sec doang sebenernya dari yg stochastic pake jit
- FYI, kalau pake yg SGD biasanya consistently ada 3 data lebih banyak yg dipredict dengan benar compared to logregnya scikit,
  but, after pake mini batch, jumlah yg bener dan persebarannya kalau dicek pake confusion matrix selalu consistently sama.
  Sooooooo idk man kan baru nyoba 1 dataset ya, gk tau kalau di dataset lain especially yg dipake di tubes ini bakal perform 
  kek gmn jd keep aja duls

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
    def __init__(self, learning_rate=0.01, n_iterations=100, threshold=0.5, mini_batch=True, batch_size=32):
        
        """
        PARAMS:
        
        learning_rate (float): Learning rate untuk gradient update
        n_iterations (int): jumlah iterasi/epoch buat training
        threshold (float, 0 < threshold < 1): if result >= threshold, maka hasil klasifikasi = true
        mini_batch (bool): kalau mau stochastic (yg diajarin di kelas) make it false
        batch_size (int): ukuran batch (kepake kalau mini_batch=True)
        
        """
        
        # inisiasi komponen
        self.learning_rate = learning_rate
        self.n_iteration = n_iterations
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.mini_batch = mini_batch
        self.batch_size = batch_size

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def fit(self, x, y):
        if self.mini_batch:
            self._fit_mbgd(x, y)
        else:
            self._fit_sgd(x, y)
    
    def _fit_sgd(self, x, y):
        "melatih model dengan stochastic gradient descent"
        
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
            
            # pake yg jit
            self.weights = sgd_update_numba(
                x_shuffeled, y_shuffeled, self.weights, 
                self.learning_rate, n_samples
            )
    
    def _fit_mbgd(self, x, y):
        "melatih model dengan mini batch gradient descent"
        
        n_samples, n_features = x.shape
        
        # Bias diinisiasi 1
        x_biased = np.hstack((np.ones((n_samples, 1)), x))
        n_total_features = x_biased.shape[1]
        
        # Initialize weights
        self.weights = np.zeros(n_total_features)
        
        # Hitung jumlah batch
        n_batches = n_samples // self.batch_size
        
        # TODO: Ubah epoch -> _
        
        for epoch in range(self.n_iteration):
            # Shuffle sekali di awal epoch
            indices = np.random.permutation(n_samples)
            x_shuffled = x_biased[indices]
            y_shuffled = y[indices]
            
            # Proses per mini-batch
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                
                # Ambil batch
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Vectorized z calc
                z = x_batch @ self.weights 
                p = self.sigmoid(z)  # Batch predictions
                
                # Vectorized gradient jadi seluruh batch sekaligus
                gradient = x_batch.T @ (y_batch - p) / self.batch_size
                
                # Update weights
                self.weights += self.learning_rate * gradient
            
            # Handle sisa yang gk masuk batch
            if n_samples % self.batch_size != 0:
                x_batch = x_shuffled[n_batches * self.batch_size:]
                y_batch = y_shuffled[n_batches * self.batch_size:]
                
                z = x_batch @ self.weights
                p = self.sigmoid(z)
                gradient = x_batch.T @ (y_batch - p) / len(x_batch)
                self.weights += self.learning_rate * gradient

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
