"""
IMPLEMENTASI K-MEANS CLUSTERING DARI AWAL
untuk Tugas Besar Pembelajaran Mesin - CAD-RADS Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, cdist
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ============================================================================
# 1. KELAS KMEANS DARI AWAL
# ============================================================================

class KMeansCustom:
    """
    Implementasi K-Means Clustering dari awal tanpa menggunakan library
    
    Parameters:
    -----------
    k : int
        Jumlah kluster
    max_iter : int
        Jumlah iterasi maksimal (default=100)
    random_state : int
        Seed untuk reprodusibilitas (default=None)
    verbose : bool
        Tampilkan informasi iterasi (default=False)
    """
    
    def __init__(self, k=3, max_iter=100, random_state=None, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.labels = None
        self.history = []  # Simpan history untuk analisis
        
    def _initialize_centroids(self, X):
        """
        Inisialisasi centroid secara random dari data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data input
            
        Returns:
        --------
        centroids : array, shape (k, n_features)
            Centroid yang sudah diinisialisasi
        """
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices].copy()
    
    def _assign_clusters(self, X):
        """
        Assign setiap data point ke centroid terdekat
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data input
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Label kluster untuk setiap data point
        """
        # Hitung jarak Euclidean dari setiap titik ke semua centroid
        distances = cdist(X, self.centroids, metric='euclidean')
        # Assign ke centroid terdekat
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """
        Update centroid dengan menghitung mean dari setiap kluster
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data input
        labels : array, shape (n_samples,)
            Label kluster saat ini
            
        Returns:
        --------
        new_centroids : array, shape (k, n_features)
            Centroid yang sudah diupdate
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            # Dapatkan semua points yang termasuk dalam kluster i
            cluster_points = X[labels == i]
            
            if len(cluster_points) > 0:
                # Update centroid sebagai mean dari points dalam kluster
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Jika cluster kosong, keep centroid yang lama
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def _calculate_sse(self, X, labels):
        """
        Hitung Sum of Squared Error (SSE) untuk mengukur kualitas clustering
        
        Parameters:
        -----------
        X : array-like
            Data input
        labels : array
            Label kluster
            
        Returns:
        --------
        sse : float
            Sum of Squared Error
        """
        sse = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                # SSE = sum((point - centroid)^2)
                sse += np.sum((cluster_points - self.centroids[i]) ** 2)
        return sse
    
    def fit(self, X):
        """
        Fit model K-Means ke data
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data training
            
        Returns:
        --------
        self : object
        """
        # Inisialisasi centroid
        self.centroids = self._initialize_centroids(X)
        
        if self.verbose:
            print(f"Inisialisasi centroid dengan k={self.k}")
            print(f"Shape data: {X.shape}")
        
        # Iterasi
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X)
            
            # Update centroids
            new_centroids = self._update_centroids(X, labels)
            
            # Hitung SSE
            sse = self._calculate_sse(X, labels)
            
            # Simpan history
            self.history.append({
                'iteration': iteration,
                'sse': sse,
                'labels': labels.copy()
            })
            
            if self.verbose:
                print(f"Iterasi {iteration + 1}/{self.max_iter} - SSE: {sse:.4f}")
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                if self.verbose:
                    print(f"Konvergen pada iterasi {iteration + 1}")
                self.labels = labels
                self.centroids = new_centroids
                break
            
            # Update centroids untuk iterasi berikutnya
            self.centroids = new_centroids
        
        self.labels = labels
        return self
    
    def predict(self, X):
        """
        Predict label kluster untuk data baru
        
        Parameters:
        -----------
        X : array-like
            Data yang akan diprediksi
            
        Returns:
        --------
        labels : array
            Label kluster
        """
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """
        Fit model dan return labels
        
        Parameters:
        -----------
        X : array-like
            Data
            
        Returns:
        --------
        labels : array
            Label kluster
        """
        self.fit(X)
        return self.labels


# ============================================================================
# 2. FUNGSI-FUNGSI UNTUK EVALUASI CLUSTERING QUALITY
# ============================================================================

def calculate_silhouette_score(X, labels):
    """
    Hitung Silhouette Score untuk mengukur kualitas cluster
    
    Silhouette Score berkisar dari -1 hingga 1:
    - 1: sampel jauh dari cluster lain
    - 0: sampel di antara dua cluster
    - -1: sampel mungkin sudah di cluster yang salah
    
    Parameters:
    -----------
    X : array-like
        Data
    labels : array
        Label kluster
        
    Returns:
    --------
    silhouette_avg : float
        Rata-rata silhouette score
    silhouette_values : array
        Silhouette score untuk setiap sampel
    """
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    unique_labels = np.unique(labels)
    
    for i in range(n_samples):
        # a(i) = mean distance dari point i ke semua points dalam cluster yang sama
        cluster_i = labels[i]
        cluster_points = X[labels == cluster_i]
        
        if len(cluster_points) > 1:
            a_i = np.mean([euclidean(X[i], p) for p in cluster_points if not np.array_equal(X[i], p)])
        else:
            a_i = 0
        
        # b(i) = minimum mean distance dari point i ke points dalam cluster lain
        b_i = np.inf
        for label in unique_labels:
            if label != cluster_i:
                other_cluster_points = X[labels == label]
                if len(other_cluster_points) > 0:
                    distance = np.mean([euclidean(X[i], p) for p in other_cluster_points])
                    b_i = min(b_i, distance)
        
        # Hitung silhouette score
        if max(a_i, b_i) == 0:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
    
    silhouette_avg = np.mean(silhouette_values)
    
    return silhouette_avg, silhouette_values


def davies_bouldin_index(X, labels, centroids):
    """
    Hitung Davies-Bouldin Index untuk mengukur kualitas cluster
    
    DB Index yang lebih rendah menunjukkan clustering yang lebih baik
    
    Parameters:
    -----------
    X : array-like
        Data
    labels : array
        Label kluster
    centroids : array
        Centroid dari setiap kluster
        
    Returns:
    --------
    db_index : float
        Davies-Bouldin Index
    """
    k = len(np.unique(labels))
    db_index = 0.0
    
    # Hitung average distance dari setiap cluster
    avg_distances = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            avg_distances[i] = np.mean([euclidean(p, centroids[i]) for p in cluster_points])
    
    # Hitung DB Index
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                centroid_distance = euclidean(centroids[i], centroids[j])
                if centroid_distance > 0:
                    ratio = (avg_distances[i] + avg_distances[j]) / centroid_distance
                    max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    db_index /= k
    
    return db_index


def elbow_method(X, k_range=range(2, 10)):
    """
    Implementasi Elbow Method untuk menentukan jumlah cluster optimal
    
    Parameters:
    -----------
    X : array-like
        Data
    k_range : range atau list
        Range jumlah cluster yang akan ditest
        
    Returns:
    --------
    sse_values : list
        SSE untuk setiap k
    silhouette_scores : list
        Silhouette score untuk setiap k
    """
    sse_values = []
    silhouette_scores = []
    db_indices = []
    
    for k in k_range:
        kmeans = KMeansCustom(k=k, random_state=42)
        labels = kmeans.fit_predict(X)
        
        # Hitung SSE
        sse = 0.0
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - kmeans.centroids[i]) ** 2)
        sse_values.append(sse)
        
        # Hitung Silhouette Score
        sil_score, _ = calculate_silhouette_score(X, labels)
        silhouette_scores.append(sil_score)
        
        # Hitung Davies-Bouldin Index
        db_index = davies_bouldin_index(X, labels, kmeans.centroids)
        db_indices.append(db_index)
        
        print(f"k={k} | SSE: {sse:.4f} | Silhouette: {sil_score:.4f} | DB Index: {db_index:.4f}")
    
    return sse_values, silhouette_scores, db_indices


# ============================================================================
# 3. FUNGSI VISUALISASI
# ============================================================================

def plot_elbow_curve(k_range, sse_values, silhouette_scores, db_indices):
    """
    Plot Elbow Curve dan metrik lainnya
    
    Parameters:
    -----------
    k_range : range
        Range jumlah cluster
    sse_values : list
        SSE untuk setiap k
    silhouette_scores : list
        Silhouette score untuk setiap k
    db_indices : list
        Davies-Bouldin Index untuk setiap k
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # SSE
    axes[0].plot(list(k_range), sse_values, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Jumlah Cluster (k)', fontsize=12)
    axes[0].set_ylabel('Sum of Squared Error (SSE)', fontsize=12)
    axes[0].set_title('Elbow Method', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Silhouette Score
    axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Jumlah Cluster (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Index
    axes[2].plot(list(k_range), db_indices, 'ro-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Jumlah Cluster (k)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index (lebih rendah = lebih baik)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_clusters_2d(X, labels, centroids, title="K-Means Clustering"):
    """
    Plot hasil clustering dalam 2D (menggunakan 2 fitur pertama)
    
    Parameters:
    -----------
    X : array-like
        Data
    labels : array
        Label kluster
    centroids : array
        Centroid dari setiap kluster
    title : str
        Judul plot
    """
    plt.figure(figsize=(10, 7))
    
    # Plot setiap cluster dengan warna berbeda
    colors = plt.cm.Set3(np.linspace(0, 1, len(np.unique(labels))))
    
    for i in np.unique(labels):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   label=f'Cluster {i}', alpha=0.6, s=50, color=colors[i])
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
               marker='X', s=300, color='red', edgecolors='black', 
               linewidths=2, label='Centroids')
    
    plt.xlabel(f'Feature 0', fontsize=12)
    plt.ylabel(f'Feature 1', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_silhouette_analysis(X, labels, k):
    """
    Plot Silhouette Analysis untuk setiap cluster
    
    Parameters:
    -----------
    X : array-like
        Data
    labels : array
        Label kluster
    k : int
        Jumlah cluster
    """
    silhouette_avg, silhouette_values = calculate_silhouette_score(X, labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_lower = 10
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    
    for i in range(k):
        cluster_silhouette_values = silhouette_values[labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, cluster_silhouette_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
                         label=f'Cluster {i}')
        
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette Coefficient', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f'Rata-rata: {silhouette_avg:.3f}')
    ax.set_title(f'Silhouette Analysis (k={k})', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. CONTOH PENGGUNAAN DENGAN DATA DUMMY
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("IMPLEMENTASI K-MEANS CLUSTERING DARI AWAL")
    print("="*70)
    
    # Generate data dummy untuk testing
    np.random.seed(42)
    # Buat 3 cluster dengan 100 points masing-masing
    X1 = np.random.randn(100, 2) + np.array([2, 2])
    X2 = np.random.randn(100, 2) + np.array([-2, 2])
    X3 = np.random.randn(100, 2) + np.array([0, -2])
    X = np.vstack([X1, X2, X3])
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n1. ELBOW METHOD - Menentukan jumlah cluster optimal")
    print("-"*70)
    sse_values, silhouette_scores, db_indices = elbow_method(X_scaled, k_range=range(2, 10))
    
    print("\n2. MEMILIH K OPTIMAL DAN FIT MODEL")
    print("-"*70)
    optimal_k = 3
    kmeans = KMeansCustom(k=optimal_k, random_state=42, verbose=True)
    labels = kmeans.fit_predict(X_scaled)
    
    print(f"\n✓ Model fitted dengan k={optimal_k}")
    print(f"  Centroid shape: {kmeans.centroids.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Jumlah iterasi: {len(kmeans.history)}")
    
    print("\n3. EVALUASI KUALITAS CLUSTERING")
    print("-"*70)
    sil_score, sil_values = calculate_silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_index(X_scaled, labels, kmeans.centroids)
    
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Index: {db_score:.4f}")
    print(f"\nInterpretasi:")
    print(f"  - Silhouette Score mendekati 1 = clustering sangat baik")
    print(f"  - DB Index lebih rendah = clustering lebih baik")
    
    print("\n4. VISUALISASI")
    print("-"*70)
    plot_clusters_2d(X_scaled, labels, kmeans.centroids, 
                    title=f"K-Means Clustering (k={optimal_k})")
    plot_elbow_curve(range(2, 10), sse_values, silhouette_scores, db_indices)
    plot_silhouette_analysis(X_scaled, labels, optimal_k)
    
    print("\n" + "="*70)
    print("SELESAI!")
    print("="*70)
