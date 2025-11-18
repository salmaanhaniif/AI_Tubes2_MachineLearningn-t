import numpy as np
from numba import jit

"""

sangat cmiiw anjay tolong di proofread lagi gue suka gk teliti wallahi -nom

NOTE: JANLUP HAPUS TES.IPYNB AND TITANIC_CLEAN.CSV KALAU UDH BISA PAKE DATASET YG RILL

Why pake numba:
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
        
        # Untuk tracking training history (bonus video)
        self.training_history = []

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _compute_loss(self, x, y):
        """Compute log loss untuk tracking"""
        n_samples = x.shape[0]
        x_biased = np.hstack((np.ones((n_samples, 1)), x))
        
        z = x_biased @ self.weights
        p = self.sigmoid(z)
        
        # Log loss (cross-entropy)
        epsilon = 1e-15  # Avoid log(0)
        p = np.clip(p, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss
    
    def fit(self, x, y, track_history=False):
        """
        Train model
        
        track_history (bool): True jika mau generate video (akan simpan weights tiap epoch)
        """
        if self.mini_batch:
            self._fit_mbgd(x, y, track_history)
        else:
            self._fit_sgd(x, y, track_history)
    
    def _fit_sgd(self, x, y, track_history=False):
        """melatih model dengan stochastic gradient descent"""
        n_samples = x.shape[0]

        # bias diinisiasi 1
        x_biased = np.hstack((np.ones((n_samples, 1)), x))
        n_total_features = x_biased.shape[1]

        self.weights = np.zeros(n_total_features)
        
        # Clear history
        if track_history:
            self.training_history = []
        
        for epoch in range(self.n_iteration):
            # urutan data dirandom 
            shuffled_indice = np.random.permutation(n_samples)
            x_shuffeled = x_biased[shuffled_indice]
            y_shuffeled = y[shuffled_indice]
            
            # pake yg jit
            self.weights = sgd_update_numba(
                x_shuffeled, y_shuffeled, self.weights, 
                self.learning_rate, n_samples
            )
            
            # Track weights dan loss
            if track_history:
                loss = self._compute_loss(x, y)
                self.training_history.append({
                    'epoch': epoch,
                    'weights': self.weights.copy(),
                    'loss': loss
                })
    
    def _fit_mbgd(self, x, y, track_history=False):
        """melatih model dengan mini batch gradient descent"""
        n_samples, n_features = x.shape
        
        # Bias diinisiasi 1
        x_biased = np.hstack((np.ones((n_samples, 1)), x))
        n_total_features = x_biased.shape[1]
        
        # Initialize weights
        self.weights = np.zeros(n_total_features)
        
        # Clear history
        if track_history:
            self.training_history = []
        
        # Hitung jumlah batch
        n_batches = n_samples // self.batch_size
        
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
            
            # Track weights dan loss
            if track_history:
                loss = self._compute_loss(x, y)
                self.training_history.append({
                    'epoch': epoch,
                    'weights': self.weights.copy(),
                    'loss': loss
                })

    def predict_probability(self, x):
        """untuk prediksi data test"""
        n_samples = x.shape[0]
        x_biased = np.hstack((np.ones((n_samples, 1)), x))

        linear_model = x_biased @ self.weights 
        probabilities = self.sigmoid(linear_model)
        return probabilities
    
    def predict(self, x):
        probabilities = self.predict_probability(x)
        predictions = (probabilities >= self.threshold).astype(int)
        return predictions
    
    def generate_training_gif(self, X, y, subsample_rate = 10, output_path='training_animation.gif', 
                             fps=10, sample_points=50):
        """
        Generate GIF that shows loss contour and parameter trajectory selama training
        
        PARAMETERS:
        
        X : training data 
        y : training labels
        subsample_rate (int): per frame mewakili x amount of sample
        output_path (str) : path to save GIF
        fps (int) : frames per second for the GIF
        sample_points (int) : jumlah sample points untuk grid loss contour
        
        
        
        
        CONTOH CARA PAKE DI NOTEBOOK:
        
        output_file = 'out.gif'

        used_model.generate_training_gif(
            X=X_train_scaled, 
            y=y_train, 
            subsample_rate=20,
            output_path=output_file, 
            fps=5              
        )

        from IPython.display import Image, display
        display(Image(filename=output_file))
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
        except ImportError:
            print("Error: matplotlib required for GIF generation, try pip install matplotlib pillow")
            return
        
        if not self.training_history:
            print("Error: No training history found. Train with track_history=True first")
            return
        
        feature_idx = (0, 1)
        
        X_subset = X[:, 0].reshape(-1, 1)
        subsampled_history = self.training_history[::subsample_rate]
        
        epochs = [h['epoch'] for h in subsampled_history]
        losses = [h['loss'] for h in subsampled_history]
        
        # Extract theta 0 & 1 sesuai spek
        theta0_vals = [h['weights'][0] for h in subsampled_history]
        theta1_vals = [h['weights'][1] for h in subsampled_history]
        
        # Create loss landscape grid
        theta0_range = np.linspace(min(theta0_vals) - 1, max(theta0_vals) + 1, sample_points)
        theta1_range = np.linspace(min(theta1_vals) - 1, max(theta1_vals) + 1, sample_points)
        theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)
        
        # Compute loss untuk setiap grid point feature 0 & 1
        loss_grid = np.zeros_like(theta0_grid)
        X_biased = np.hstack((np.ones((X_subset.shape[0], 1)), X_subset))
        
        print("Computing loss landscape...")
        for i in range(sample_points):
            for j in range(sample_points):
                temp_weights = np.zeros(X_biased.shape[1])
                temp_weights[0] = theta0_grid[i, j]
                temp_weights[1] = theta1_grid[i, j]
                
                z = X_biased @ temp_weights
                p = self.sigmoid(z)
                epsilon = 1e-15
                p = np.clip(p, epsilon, 1 - epsilon)
                loss_grid[i, j] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        
        # Create animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            # Plot 1: Loss contour dengan trajectory
            contour = ax1.contour(theta0_grid, theta1_grid, loss_grid, levels=20, cmap='viridis', alpha=0.6)
            ax1.clabel(contour, inline=True, fontsize=8)
            
            # Plot trajectory sampai frame saat ini
            ax1.plot(theta0_vals[:frame+1], theta1_vals[:frame+1], 
                    'r-', linewidth=2, label='Parameter trajectory')
            ax1.plot(theta0_vals[frame], theta1_vals[frame], 
                    'ro', markersize=10, label=f'Epoch {epochs[frame]}')
            
            ax1.set_xlabel('θ₀ (bias)')
            ax1.set_ylabel(f'θ₁ (weight feature {feature_idx[0]})')
            ax1.set_title(f'Loss Contour & Parameter Trajectory\nEpoch: {epochs[frame]}, Loss: {losses[frame]:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Loss over time
            ax2.plot(epochs[:frame+1], losses[:frame+1], 'b-', linewidth=2)
            ax2.plot(epochs[frame], losses[frame], 'ro', markersize=10)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss Over Time')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, max(epochs))
            ax2.set_ylim(0, max(losses) * 1.1)
        
        print(f"Generating GIF with {len(self.training_history)} frames...")
        anim = FuncAnimation(fig, animate, frames=len(subsampled_history), 
                           interval=1000//fps, repeat=True)
        
        # Save GIF
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()
        
        print(f"GIF saved to {output_path}")
        print(f"Duration: {len(self.training_history)/fps:.1f} seconds")