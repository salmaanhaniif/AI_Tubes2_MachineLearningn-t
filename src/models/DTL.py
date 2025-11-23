import numpy as np
from collections import Counter
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Node:
    def __init__(self, feature=None, threshold=None, categories=None, is_nominal=False, children=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.categories = categories
        self.is_nominal = is_nominal
        self.children = children if children is not None else {}
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeLearning:
    def __init__(self, min_samples_split=2, max_depth=100, feature_types=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.feature_types = feature_types 
        self.root = None
        
    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p))
        
    def _information_gain(self, parent_y, children_y):
        parent_entropy = self._entropy(parent_y)
        children_entropy = 0
        total_len = len(parent_y)
        
        for child in children_y:
            if len(child) > 0:
                children_entropy += (len(child) / total_len) * self._entropy(child)
        return parent_entropy - children_entropy

    def _split_information(self, children_y_list):
        total = sum(len(child) for child in children_y_list)
        split_info = 0
        for child in children_y_list:
            if len(child) > 0:
                p = len(child) / total
                split_info += p * np.log2(p) 
        return -split_info
            
    def _gain_ratio(self, parent_y, children_y_list):
        split_info = self._split_information(children_y_list)
        info_gain = self._information_gain(parent_y, children_y_list)
        
        return info_gain / split_info if split_info != 0 else 0

    def _is_missing(self, value):
        if value is None: return True
        if isinstance(value, float) and np.isnan(value): 
            return True
        if isinstance(value, str) and value.lower() in ['nan', 'none', '', '?', 'na', 'n/a']: 
            return True
        return False
    
    def _determine_feature_types(self, X):
        feature_types = []
        n_features = X.shape[1]
        
        for i in range(n_features):
            col_values = X[:, i]
            valid_vals = [v for v in col_values if not self._is_missing(v)]
                
            is_numerical = True
            for val in valid_vals:
                if isinstance(val, str) or isinstance(val, bool):
                    is_numerical = False
                    break    
            feature_types.append('numerical' if is_numerical else 'nominal')          
        return feature_types

    def _find_best_numerical_split(self, X, y, feature_idx):
        feature_values = X[:, feature_idx] 
        
        non_missing_indices = [i for i, v in enumerate(feature_values) if not self._is_missing(v)]
        
        if len(non_missing_indices) < 2:
            return None, -1

        non_missing_vals = [float(feature_values[i]) for i in non_missing_indices]
        non_missing_y = [y[i] for i in non_missing_indices]
        
        unique_values = sorted(set(non_missing_vals))
        
        if len(unique_values) <= 1:
            return None, -1
        
        thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(len(unique_values) - 1)]
        
        best_threshold = None
        best_gain_ratio = -1
        
        # numpy array untuk optimasi
        nm_vals_np = np.array(non_missing_vals)
        nm_y_np = np.array(non_missing_y)

        for t in thresholds:
            left_mask = nm_vals_np <= t
            right_mask = ~left_mask
            
            left_y = nm_y_np[left_mask]
            right_y = nm_y_np[right_mask]
            
            if len(left_y) == 0 or len(right_y) == 0:
                continue

            current_gain = self._gain_ratio(non_missing_y, [left_y, right_y])
            
            if current_gain > best_gain_ratio:
                best_gain_ratio = current_gain
                best_threshold = t
        
        return best_threshold, best_gain_ratio
            
    def _find_best_nominal_split(self, X, y, feature_idx):
        feature_values = X[:, feature_idx] if isinstance(X, np.ndarray) else [row[feature_idx] for row in X]
        
        non_missing_indices = [i for i, v in enumerate(feature_values) if not self._is_missing(v)]
        
        if not non_missing_indices:
            return None, -1
            
        non_missing_vals = [feature_values[i] for i in non_missing_indices]
        non_missing_y = [y[i] for i in non_missing_indices]
        
        unique_categories = list(set(non_missing_vals))
        if len(unique_categories) <= 1:
            return None, -1
        
        children_y_list = []
        nm_vals_np = np.array(non_missing_vals)
        nm_y_np = np.array(non_missing_y)
        
        for category in unique_categories:
            mask = nm_vals_np == category
            children_y_list.append(nm_y_np[mask])
        
        gain_ratio = self._gain_ratio(non_missing_y, children_y_list)
        return unique_categories, gain_ratio            
    
    def _find_best_split(self, X, y):
        max_gain_ratio = -1
        best_feature_idx = None
        best_split_info = None 
        
        n_features = X.shape[1]
        
        for idx in range(n_features):
            feature_type = self.feature_types[idx] if self.feature_types else 'numerical'
            
            if feature_type == 'nominal':
                split_info, gain_ratio = self._find_best_nominal_split(X, y, idx)
            else:
                split_info, gain_ratio = self._find_best_numerical_split(X, y, idx)
            
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_feature_idx = idx
                best_split_info = split_info
                
        return best_feature_idx, best_split_info, max_gain_ratio
        
    def _create_leaf(self, y):
        counter_y = Counter(y)
        majority_class = counter_y.most_common(1)[0][0]
        return Node(value=majority_class)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return self._create_leaf(y)
        if n_labels == 1:
            return Node(value=y[0])
            
        best_feature, split_info, gain_ratio = self._find_best_split(X, y)
        
        if best_feature is None or gain_ratio <= 0:
            return self._create_leaf(y)
        
        node = Node(feature=best_feature)
        f_type = self.feature_types[best_feature] if self.feature_types else 'numerical'
        
        if f_type == 'numerical':
            node.threshold = split_info
            node.is_nominal = False
            
            col_vals = X[:, best_feature]
            
            left_indices = []
            right_indices = []
            
            for i, v in enumerate(col_vals):
                if self._is_missing(v):
                    continue
                if float(v) <= node.threshold:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            
            missing_indices = [i for i, v in enumerate(col_vals) if self._is_missing(v)]
            if len(left_indices) >= len(right_indices):
                left_indices.extend(missing_indices)
            else:
                right_indices.extend(missing_indices)
                
            if not left_indices or not right_indices:
                return self._create_leaf(y)
                
            node.children['left'] = self._build_tree(X[left_indices], y[left_indices], depth+1)
            node.children['right'] = self._build_tree(X[right_indices], y[right_indices], depth+1)
            
        else: # nominal
            node.categories = split_info
            node.is_nominal = True
            
            idx_map = {cat: [] for cat in node.categories}
            missing_indices = []
            
            for i, v in enumerate(X[:, best_feature]):
                if self._is_missing(v):
                    missing_indices.append(i)
                elif v in idx_map:
                    idx_map[v].append(i)
                    
            counter_y = Counter(y)
            majority_class = counter_y.most_common(1)[0][0]
            
            idx_map[majority_class].extend(missing_indices)
            
            for category in node.categories:
                indices = idx_map[category]
                if not indices:
                    node.children[category] = self._create_leaf(y)
                else:
                    node.children[category] = self._build_tree(X[indices], y[indices], depth+1)
        return node
        
    def fit(self, X, y):
        X = np.array(X, dtype=object) 
        y = np.array(y)
        if self.feature_types is None:
            self.feature_types = self._determine_feature_types(X)
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, sample, node):
        if node.is_leaf_node():
            return node.value
            
        val = sample[node.feature]
        
        if self._is_missing(val):
            first_child = list(node.children.values())[0]
            return self._traverse_tree(sample, first_child)

        if node.is_nominal:
            if val in node.children:
                return self._traverse_tree(sample, node.children[val])
            return self._traverse_tree(sample, list(node.children.values())[0])
        else:
            if val <= node.threshold:
                return self._traverse_tree(sample, node.children['left'])
            else:
                return self._traverse_tree(sample, node.children['right'])

    def predict(self, X):
        return np.array([self._traverse_tree(sample, self.root) for sample in X])
    
# --- Testing Code ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    csv_path = os.path.join(base_dir, 'src/SeoulBikeData.csv')
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='euc-kr')
        
    X = df.iloc[:, :-1].values.tolist()  
    y = df.iloc[:, -1].values.tolist()   
    
    print("Data dari CSV berhasil dimuat!")
    print(f"Total data: {len(X)}")
    print(f"Jumlah fitur: {len(X[0])}")
    
    X = df.iloc[:, :-1].values.tolist()  
    y = df.iloc[:, -1].values.tolist()   
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data training: {len(X_train)}")
    print(f"Data testing: {len(X_test)}\n")
    
    clf = DecisionTreeLearning(min_samples_split=2, max_depth=10)
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    train_accuracy = np.mean(y_train_pred == np.array(y_train))
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    y_test_pred = clf.predict(X_test)
    test_accuracy = np.mean(y_test_pred == np.array(y_test))
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    print(f"\nSample prediksi 5 data test pertama:")
    for i in range(min(5, len(X_test))):
        print(f"Data: {X_test[i]}")
        print(f"Prediksi: {y_test_pred[i]}, Actual: {y_test[i]}\n")
        
    # Contoh Data (Campuran Numerik dan Nominal)
    # [Cuaca, Suhu, Kelembaban, Angin]
    # X = [
    #     ['Sunny', 85, 85, 'Weak'],
    #     ['Sunny', 80, 90, 'Strong'],
    #     ['Overcast', 83, 86, 'Weak'],
    #     ['Rain', 70, 96, 'Weak'],
    #     ['Rain', 68, 80, 'Weak'],
    #     ['Rain', 65, 70, 'Strong'],
    #     ['Overcast', 64, 65, 'Strong'],
    #     ['Sunny', 72, 95, 'Weak'],
    #     ['Sunny', 69, 70, 'Weak'],
    #     ['Rain', 75, 80, 'Weak'],
    #     ['Sunny', 75, 70, 'Strong'],
    #     ['Overcast', 72, 90, 'Strong'],
    #     ['Overcast', 81, 75, 'Weak'],
    #     ['Rain', 71, 91, 'Strong']
    # ]
    # y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
    
    # # Definisikan tipe fitur secara manual    
    # clf = DecisionTreeLearning(min_samples_split=2, max_depth=5)
    # clf.fit(X, y)
    
    # print("Prediksi data latih:")
    # print(clf.predict(X))
    
    # # Test data baru
    # X_test = [['Sunny', 85, 85, 'Weak'], ['Rain', 70, 96, 'Weak']]
    # print("Prediksi data baru:")
    # print(clf.predict(X_test))