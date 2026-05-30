"""
====================================================================
BAGIAN 2: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
====================================================================
Lanjutan dari Supervised Learning section
Dataset: CAD-RADS (CADalizadeh.xls)
"""

print("\n" + "="*70)
print("BAGIAN 6: UNSUPERVISED LEARNING - K-MEANS CLUSTERING")
print("="*70)

# ============================================================================
# STEP 1: IMPLEMENTASI K-MEANS DARI AWAL (TANPA SKLEARN)
# ============================================================================

print("\n[STEP 1] Implementasi K-Means Algorithm dari awal...")

from scipy.spatial.distance import euclidean, cdist

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
        Seed untuk reproducibility (default=None)
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
        self.history = []
        
    def _initialize_centroids(self, X):
        """Inisialisasi centroid secara random dari data"""
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices].copy()
    
    def _assign_clusters(self, X):
        """Assign setiap data point ke centroid terdekat"""
        distances = cdist(X, self.centroids, metric='euclidean')
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        """Update centroid dengan menghitung mean dari setiap kluster"""
        new_centroids = np.zeros((self.k, X.shape[1]))
        
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        
        return new_centroids
    
    def _calculate_sse(self, X, labels):
        """Hitung Sum of Squared Error (SSE)"""
        sse = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - self.centroids[i]) ** 2)
        return sse
    
    def fit(self, X):
        """Fit model K-Means ke data"""
        self.centroids = self._initialize_centroids(X)
        
        if self.verbose:
            print(f"  Inisialisasi centroid dengan k={self.k}")
        
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
                print(f"  Iterasi {iteration + 1}/{self.max_iter} - SSE: {sse:.4f}")
            
            # Check convergence
            if np.allclose(self.centroids, new_centroids):
                if self.verbose:
                    print(f"  ✓ Konvergen pada iterasi {iteration + 1}")
                self.labels = labels
                self.centroids = new_centroids
                break
            
            self.centroids = new_centroids
        
        self.labels = labels
        return self
    
    def predict(self, X):
        """Predict label kluster untuk data baru"""
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        """Fit model dan return labels"""
        self.fit(X)
        return self.labels


print("✓ Class KMeansCustom sudah didefinisikan")

# ============================================================================
# STEP 2: FUNGSI-FUNGSI EVALUASI CLUSTERING QUALITY
# ============================================================================

print("\n[STEP 2] Definisikan fungsi evaluasi clustering quality...")

def calculate_silhouette_score(X, labels):
    """
    Hitung Silhouette Score
    Range: -1 hingga 1 (semakin tinggi semakin baik)
    """
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)
    
    unique_labels = np.unique(labels)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        cluster_points = X[labels == cluster_i]
        
        # a(i) = mean distance dari point i ke semua points dalam cluster yang sama
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
    Hitung Davies-Bouldin Index
    Semakin rendah semakin baik
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


print("✓ Fungsi evaluasi sudah didefinisikan")

# ============================================================================
# STEP 3: PERSIAPAN DATA UNTUK UNSUPERVISED LEARNING
# ============================================================================

print("\n[STEP 3] Persiapan data untuk Unsupervised Learning...")

# Gunakan X_train_s yang sudah di-normalize dari Supervised Learning section
# Tapi kali ini kita gunakan seluruh data yang sudah di-scale

X_unsupervised = X_train_s.copy()

print(f"  Data shape: {X_unsupervised.shape}")
print(f"  Fitur yang digunakan: {X.columns.tolist()}")
print(f"✓ Data sudah siap untuk clustering")

# ============================================================================
# STEP 4: ELBOW METHOD - MENENTUKAN K OPTIMAL
# ============================================================================

print("\n[STEP 4] Elbow Method - Menentukan jumlah cluster optimal...")
print("  (Ini mungkin memakan waktu beberapa detik...)")

k_range = range(2, 11)
sse_values = []
silhouette_scores = []
db_indices = []

for k in k_range:
    print(f"  Testing k={k}...", end=" ")
    
    kmeans = KMeansCustom(k=k, random_state=42, verbose=False)
    labels = kmeans.fit_predict(X_unsupervised)
    
    # SSE
    sse = 0.0
    for i in range(k):
        cluster_points = X_unsupervised[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - kmeans.centroids[i]) ** 2)
    sse_values.append(sse)
    
    # Silhouette Score
    sil_score, _ = calculate_silhouette_score(X_unsupervised, labels)
    silhouette_scores.append(sil_score)
    
    # Davies-Bouldin Index
    db_index = davies_bouldin_index(X_unsupervised, labels, kmeans.centroids)
    db_indices.append(db_index)
    
    print(f"SSE={sse:.2f}, Silhouette={sil_score:.4f}, DB={db_index:.4f}")

print("✓ Elbow method selesai")

# ============================================================================
# STEP 5: VISUALISASI ELBOW METHOD
# ============================================================================

print("\n[STEP 5] Visualisasi Elbow Method...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# SSE
axes[0].plot(list(k_range), sse_values, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Sum of Squared Error (SSE)', fontsize=12, fontweight='bold')
axes[0].set_title('Elbow Method - SSE', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(k_range))

# Silhouette Score
axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(k_range))
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

# Davies-Bouldin Index
axes[2].plot(list(k_range), db_indices, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[2].set_title('Davies-Bouldin Index (lebih rendah = lebih baik)', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(list(k_range))

plt.tight_layout()
plt.show()

# Print tabel perbandingan
print("\nTabel Perbandingan Metrik untuk berbagai K:")
print("-" * 70)
comparison_df = pd.DataFrame({
    'K': list(k_range),
    'SSE': [f"{x:.2f}" for x in sse_values],
    'Silhouette Score': [f"{x:.4f}" for x in silhouette_scores],
    'Davies-Bouldin Index': [f"{x:.4f}" for x in db_indices]
})
print(comparison_df.to_string(index=False))
print("-" * 70)

# ============================================================================
# STEP 6: ANALISIS DAN PEMILIHAN K OPTIMAL
# ============================================================================

print("\n[STEP 6] Analisis dan pemilihan K optimal...")

# Cari optimal k berdasarkan beberapa kriteria
best_silhouette_k = list(k_range)[np.argmax(silhouette_scores)]
best_db_k = list(k_range)[np.argmin(db_indices)]

print(f"\nAnalisis metrik:")
print(f"  ✓ K dengan Silhouette Score tertinggi: k={best_silhouette_k} (score={max(silhouette_scores):.4f})")
print(f"  ✓ K dengan DB Index terendah: k={best_db_k} (index={min(db_indices):.4f})")

# Pemilihan K - bisa dilihat dari elbow point di SSE atau kombinasi metrik
print(f"\nAlasan pemilihan K:")
print(f"  1. Lihat plot SSE - cari 'elbow point' (titik di mana SSE mulai stabil)")
print(f"  2. Lihat Silhouette Score - semakin tinggi semakin baik (ideally > 0.5)")
print(f"  3. Lihat DB Index - semakin rendah semakin baik (ideally < 1.0)")

# PILIH K OPTIMAL (biasanya antara 3-5 untuk data CAD-RADS)
# Anda bisa adjust sesuai dengan analisis visual dari plot di atas
optimal_k = 3  # ← SESUAIKAN BERDASARKAN ANALISIS ANDA

print(f"\n🎯 K OPTIMAL YANG DIPILIH: {optimal_k}")
print(f"  - SSE: {sse_values[optimal_k-2]:.2f}")
print(f"  - Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}")
print(f"  - Davies-Bouldin Index: {db_indices[optimal_k-2]:.4f}")

# ============================================================================
# STEP 7: TRAINING K-MEANS DENGAN K OPTIMAL
# ============================================================================

print(f"\n[STEP 7] Training K-Means dengan k={optimal_k}...")

kmeans = KMeansCustom(k=optimal_k, max_iter=100, random_state=42, verbose=True)
cluster_labels = kmeans.fit_predict(X_unsupervised)

print(f"✓ Model training selesai!")
print(f"  - Total iterasi hingga konvergen: {len(kmeans.history)}")
print(f"  - Final SSE: {kmeans.history[-1]['sse']:.4f}")

# ============================================================================
# STEP 8: EVALUASI HASIL CLUSTERING
# ============================================================================

print(f"\n[STEP 8] Evaluasi hasil clustering...")

# Hitung metrik kualitas
silhouette_avg, silhouette_values = calculate_silhouette_score(X_unsupervised, cluster_labels)
db_score = davies_bouldin_index(X_unsupervised, cluster_labels, kmeans.centroids)

print(f"\nKualitas Clustering:")
print(f"  Silhouette Score: {silhouette_avg:.4f}")
if silhouette_avg > 0.7:
    print(f"    └─ Interpretasi: Sangat Baik ✓✓✓")
elif silhouette_avg > 0.5:
    print(f"    └─ Interpretasi: Baik ✓✓")
elif silhouette_avg > 0.25:
    print(f"    └─ Interpretasi: Cukup ✓")
else:
    print(f"    └─ Interpretasi: Kurang baik")

print(f"\n  Davies-Bouldin Index: {db_score:.4f}")
if db_score < 0.5:
    print(f"    └─ Interpretasi: Sangat Baik ✓✓✓")
elif db_score < 1.0:
    print(f"    └─ Interpretasi: Baik ✓✓")
elif db_score < 1.5:
    print(f"    └─ Interpretasi: Cukup ✓")
else:
    print(f"    └─ Interpretasi: Kurang baik")

# Distribusi cluster
print(f"\nDistribusi anggota cluster:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

# ============================================================================
# STEP 9: VISUALISASI HASIL CLUSTERING
# ============================================================================

print(f"\n[STEP 9] Visualisasi hasil clustering...")

# Plot 1: 2D Clustering (menggunakan 2 fitur pertama)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot clustering 2D
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for i in np.unique(cluster_labels):
    cluster_points = X_unsupervised[cluster_labels == i]
    axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   label=f'Cluster {i}', alpha=0.6, s=50, color=colors[i])

axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
               marker='X', s=300, color='red', edgecolors='black', 
               linewidths=2, label='Centroids', zorder=5)

axes[0].set_xlabel(f'Feature 0 ({X.columns[0]})', fontsize=11)
axes[0].set_ylabel(f'Feature 1 ({X.columns[1]})', fontsize=11)
axes[0].set_title(f'K-Means Clustering (k={optimal_k})', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Cluster Size Distribution
unique, counts = np.unique(cluster_labels, return_counts=True)
axes[1].bar([f'Cluster {c}' for c in unique], counts, color=colors, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Number of Points', fontsize=11, fontweight='bold')
axes[1].set_title(f'Cluster Size Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

for i, (c, count) in enumerate(zip(unique, counts)):
    percentage = (count / len(cluster_labels)) * 100
    axes[1].text(i, count + 2, f'{percentage:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 10: ANALISIS KARAKTERISTIK SETIAP CLUSTER
# ============================================================================

print(f"\n[STEP 10] Analisis karakteristik setiap cluster...")

# Denormalisasi centroid untuk interpretasi
centroids_original = scaler.inverse_transform(kmeans.centroids)

# Buat dataframe untuk analisis
cluster_analysis = []

for cluster_id in range(optimal_k):
    cluster_mask = cluster_labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    cluster_data = X_unsupervised[cluster_mask]
    
    cluster_info = {'Cluster': cluster_id, 'Size': cluster_size, 'Percentage': f"{cluster_size/len(cluster_labels)*100:.1f}%"}
    
    # Tambah mean nilai setiap fitur
    for feat_idx, feat_name in enumerate(X.columns):
        cluster_info[feat_name] = cluster_data[:, feat_idx].mean()
    
    cluster_analysis.append(cluster_info)

cluster_analysis_df = pd.DataFrame(cluster_analysis)

print("\nTabel Karakteristik Setiap Cluster (normalized values):")
print("-" * 100)
print(cluster_analysis_df.to_string(index=False))
print("-" * 100)

# Denormalisasi untuk interpretasi
print("\nKarakteristik Setiap Cluster (original scale values):")
print("-" * 100)

for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id}:")
    cluster_mask = cluster_labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    
    print(f"  Ukuran: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.1f}%)")
    print(f"  Centroid (original values):")
    
    for i, feature in enumerate(X.columns):
        print(f"    {feature}: {centroids_original[cluster_id, i]:.4f}")
    
    # Karakteristik cluster
    cluster_data_original = X_unsupervised[cluster_mask]
    print(f"  Karakteristik cluster:")
    
    for i, feature in enumerate(X.columns):
        mean_val = cluster_data_original[:, i].mean()
        std_val = cluster_data_original[:, i].std()
        print(f"    {feature}: mean={mean_val:.4f}, std={std_val:.4f}")

# ============================================================================
# STEP 11: SILHOUETTE PLOT
# ============================================================================

print(f"\n[STEP 11] Visualisasi Silhouette Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

y_lower = 10
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

for i in range(optimal_k):
    cluster_silhouette_values = silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_values,
                     facecolor=colors[i], edgecolor=colors[i], alpha=0.7,
                     label=f'Cluster {i}')
    
    y_lower = y_upper + 10

ax.set_xlabel('Silhouette Coefficient', fontsize=11, fontweight='bold')
ax.set_ylabel('Cluster', fontsize=11, fontweight='bold')
ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
           label=f'Rata-rata: {silhouette_avg:.3f}')
ax.set_title(f'Silhouette Analysis (k={optimal_k})', fontsize=12, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"✓ Silhouette plot ditampilkan")

# ============================================================================
# STEP 12: SIMPAN HASIL CLUSTERING
# ============================================================================

print(f"\n[STEP 12] Simpan hasil clustering...")

# Tambahkan cluster label ke dataframe original
df_clustering = X.copy()
df_clustering['Cluster'] = cluster_labels
df_clustering['Silhouette'] = silhouette_values

print(f"\nRingkasan Hasil Clustering:")
print(f"  ✓ Optimal K: {optimal_k}")
print(f"  ✓ Silhouette Score: {silhouette_avg:.4f}")
print(f"  ✓ Davies-Bouldin Index: {db_score:.4f}")
print(f"  ✓ Total samples: {len(cluster_labels)}")
print(f"  ✓ Total features: {X.shape[1]}")

# ============================================================================
# STEP 13: RESUME DAN KESIMPULAN
# ============================================================================

print("\n" + "="*70)
print("KESIMPULAN K-MEANS CLUSTERING")
print("="*70)

print(f"""
Hasil Clustering:
  - Optimal K: {optimal_k}
  - Silhouette Score: {silhouette_avg:.4f}
  - Davies-Bouldin Index: {db_score:.4f}
  
Interpretasi:
  - Silhouette Score {silhouette_avg:.4f}: {'Sangat Baik (>0.7)' if silhouette_avg > 0.7 else 'Baik (0.5-0.7)' if silhouette_avg > 0.5 else 'Cukup (0.25-0.5)' if silhouette_avg > 0.25 else 'Kurang Baik (<0.25)'}
  - DB Index {db_score:.4f}: {'Sangat Baik (<0.5)' if db_score < 0.5 else 'Baik (0.5-1.0)' if db_score < 1.0 else 'Cukup (1.0-1.5)' if db_score < 1.5 else 'Kurang Baik (>1.5)'}
  
Karakteristik Cluster:
  - Cluster 0: {counts[0]} points ({counts[0]/len(cluster_labels)*100:.1f}%)
  - Cluster 1: {counts[1]} points ({counts[1]/len(cluster_labels)*100:.1f}%)
  - Cluster 2: {counts[2] if optimal_k > 2 else 0} points ({(counts[2]/len(cluster_labels)*100) if optimal_k > 2 else 0:.1f}%)
  {'- Cluster 3: '+str(counts[3])+' points ('+f'{counts[3]/len(cluster_labels)*100:.1f}%'+')' if optimal_k > 3 else ''}
  
Insights:
  1. Data berhasil di-cluster menjadi {optimal_k} grup yang distinct
  2. Silhouette score menunjukkan kualitas clustering {'SANGAT BAIK' if silhouette_avg > 0.7 else 'BAIK' if silhouette_avg > 0.5 else 'CUKUP'}
  3. Cluster memiliki karakteristik yang berbeda berdasarkan nilai fitur
  4. Hasil clustering dapat digunakan untuk segmentasi pasien CAD-RADS
""")

print("="*70)
print("✓ BAGIAN UNSUPERVISED LEARNING - K-MEANS SELESAI")
print("="*70)
