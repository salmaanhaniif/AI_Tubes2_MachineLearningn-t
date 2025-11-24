import pickle
import numpy as np

class BinarySVM:
    # implementasi menggunakan optimasi kuadratik

    def __init__(self, C=1.0, kernel='linear', kernel_param=1.0, tol=1e-3, max_iter=100):
        self.C = C
        self.kernel_type = kernel
        self.kernel_param = kernel_param
        self.tol = tol 
        self.max_iter = max_iter 

        # parameter optimasi
        self.alphas = None # pengali langrange
        self.support_vectors = None # supprot vectors
        self.sv_labels = None
        self.sv_indices = None

        # hyperplane
        self.w = None
        self.b = None

        # chace error
        self.errors = None
        self.K = None

    def kernel(self, x1, x2):
        # menghitung kernel matrix

        # tiap kernel Kij adalah hasil dot product dari Xi . Xj (linear)
        if self.kernel_type == 'linear':
            return np.dot(x1, x2.T)
        
        elif self.kernel_type == 'polynomial':
            d = self.kernel_param
            return (np.dot(x1, x2.T) + 1) ** d
        
        elif self.kernel_type == 'rbf':
            gamma = self.kernel_param
            if x1.ndim == 1:
                x1 = x1.reshape(1, -1)
            if x2.ndim == 1:
                x2 = x2.reshape(1, -1)

            xx = np.sum(x1**2, axis=1).reshape(-1, 1)
            yy = np.sum(x2**2, axis=1).reshape(-1, 1)
            xy = np.dot(x1, x2.T)
            sq_dist = xx + yy - 2*xy
            return np.exp(-gamma * sq_dist)
        
        elif self.kernel_type == 'sigmoid':
            return np.tanh(np.dot(x1, x2.T) + 1)
        
        else:
            raise ValueError(f"Kernel '{self.kernel_type}' tidak didukung")
    
    def compute_error(self, i):
        f_xi = np.sum(self.alphas * self.y * self.K[i, :]) + self.b
        return f_xi - self.y[i]
    
    def select_second_alpha(self, i1):
        # pilih alpha yang memaksimalkan |E_i1 - E_i2| untuk konvergensi lebih cepat

        E1 = self.errors[i1]
        
        # cari i2 yang memaksimalkan |E1-E2|
        valid_alphas = np.where((self.alphas > self.tol) & (self.alphas < self.C - self.tol))[0]

        if len(valid_alphas) > 1:
            max_diff = 0
            i2 = -1
            for i in valid_alphas:
                if i == i1:
                    continue
                E2 = self.errors[i]
                diff = abs(E1-E2)
                if diff > max_diff:
                    max_diff = diffi2 = i
            if i2 >= 0:
                return i2
            
        # jika tidak ada atau cuma ada 1, pilih random
        non_i1 = list(range(len(self.y)))
        non_i1.remove(i1)
        return np.random.choice(non_i1)
    
    # implementasi SMO
    def take_step(self, i1, i2):
        # constraint: sum y_i * alpha_i = 0
        # alpha: y1*alpha1 + y2*alpha2 = constant
        
        # 1 jika berhasil update, 0 jika tidak
        if i1 == i2:
            return 0
        
        alpha1_old = self.alphas[i1]
        alpha2_old = self.alphas[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]

        if y1 != y2:
            L = max(0, alpha2_old - alpha1_old)
            H = min(self.C, self.C + alpha2_old - alpha1_old)
        else:
            L = max(0, alpha1_old + alpha2_old - self.C)
            H = min(self.C, alpha1_old + alpha2_old)
        
        if abs(L - H) < 1e-10:
            return 0
        
        #  hitung eta
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = 2 * k12 - k11 - k22
        if eta >= 0:
            return 0
        
        # hitung alpha2 new
        alpha2_new = alpha2_old - (y2 * (E1 - E2)) / eta
        if alpha2_new > H:
            alpha2_new = H
        elif alpha2_new < L:
            alpha2_new = L
        if abs(alpha2_new - alpha2_old) < 1e-5:
            return 0
        
        #  hitung alpha 1 new
        alpha1_new = alpha1_old + y1 * y2 * (alpha2_old - alpha2_new)
        b1 = self.b - E1 - y1 * (alpha1_new - alpha1_old) * k11 - y2 * (alpha2_new - alpha2_old) * k12
          
        b2 = self.b - E2 - y1 * (alpha1_new - alpha1_old) * k12 - y2 * (alpha2_new - alpha2_old) * k22
        if 0 < alpha1_new < self.C:
            self.b = b1
        elif 0 < alpha2_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

        # update alphas
        self.alphas[i1] = alpha1_new
        self.alphas[i2] = alpha2_new

        # Hanya update error untuk non-bound alphas
        for i in range(len(self.y)):
            if 0 < self.alphas[i] < self.C:
                self.errors[i] = self.compute_error(i)

        self.errors[i1] = 0 
        self.errors[i2] = 0
        
        return 1


    def examine_example(self, i2):
        #  periksa apakah alpha i2 melanggar KKT
        y2 = self.y[i2]
        alpha2 = self.alphas[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2

        if ((r2 < -self.tol and alpha2 < self.C) or 
            (r2 > self.tol and alpha2 > 0)):
            
            i1 = self.select_second_alpha(i2)
            
            if self.take_step(i1, i2):
                return 1
        
        return 0
    
    def solve_qp_problem(self):
        n_samples = len(self.y)
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        
        #  error cache
        self.errors = -self.y.copy()

        # main SMO loop
        num_changed = 0
        examine_all = True 
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            if examine_all:
                # loop semua data points
                for i in range(n_samples):
                    num_changed += self.examine_example(i)
            else:
                # loop hanya non-bound alphas
                # ini adalah support vectors aktif
                non_bound_idx = np.where((self.alphas > self.tol) & (self.alphas < self.C - self.tol))[0]
                for i in non_bound_idx:
                    num_changed += self.examine_example(i)
            
            iteration += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        return self.alphas

    def fit(self, X, y):
        # melatih SVM menggunakan SMO algo       
     
        # simpan training data
        self.X_train = X
        self.y = np.where(y <= 0, -1, 1).astype(float)

        self.K = self.kernel(X, X)
        alphas = self.solve_qp_problem()

        sv_indices = alphas > 1e-5
        self.sv_indices = np.where(sv_indices)[0]
        self.alphas = alphas[sv_indices]
        self.support_vectors = X[sv_indices]
        self.sv_labels = self.y[sv_indices]
        
        if self.kernel_type == 'linear':
            # w = sum(alpha_i * y_i * x_i)
            self.w = np.sum(
                self.alphas[:, np.newaxis] * 
                self.sv_labels[:, np.newaxis] * 
                self.support_vectors, 
                axis=0
            )
            print(f"Weight vector w: {self.w}")
        else:
            self.w = None  

        sv_predictions = np.sum(self.alphas * self.sv_labels * self.K[self.sv_indices][:, sv_indices], axis=1)
        self.b = np.mean(self.sv_labels - sv_predictions)

    def predict_score(self, X):
        if self.kernel_type == 'linear' and self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            # f(x) = sum(alpha_i * y_i * K(x, x_i)) + b
            K_test = self.kernel(X, self.support_vectors)
            scores = np.sum(self.alphas * self.sv_labels * K_test, axis=1) + self.b
            return scores

    def predict(self, X):
        # mengembalikan label kelas biner {-1, +1}.
        # klasifikasi: sign(f(x))

        scores = self.predict_score(X)
        return np.sign(scores)
    
class MulticlassSVM:
    # implementasi One v All (OvA) menggunakan BinarySVM
    def __init__(self, C=1.0, kernel='linear', kernel_param=1.0, tol=1e-3, max_iter=100):
        self.C = C
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.tol = tol
        self.max_iter = max_iter
        self.models = []
        self.classes = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        self.classes = np.unique(y)
        self.models = [] 
        
        # debug
        # print(f"Training Multiclass SVM (One against All) for {len(self.classes)} classes")
        # print(f"Mode: SMO Optimization | Kernel: {self.kernel}")

        for class_label in self.classes:
            # debug
            # print(f" -> Training model for '{class_label}' class")
            
            y_binary = np.where(y == class_label, 1, -1) # Target=1, Rest=-1
            model = BinarySVM(
                C=self.C, 
                kernel=self.kernel,
                kernel_param=self.kernel_param,
                tol=self.tol,
                max_iter=self.max_iter
            )
            
            model.fit(X, y_binary)
            self.models.append(model)
        # print("All models trained successfully.")

    def predict(self, X):
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))
        
        for i, model in enumerate(self.models):
            scores[:, i] = model.predict_score(X)
        
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model berhasil disimpan ke {filename}")

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model berhasil dimuat dari {filename}")
        return model