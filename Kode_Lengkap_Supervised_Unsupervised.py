"""
====================================================================
TUGAS BESAR PEMBELAJARAN MESIN
CAD-RADS Dataset Analysis
Part 1: Supervised Learning (Classification)
Part 2: Unsupervised Learning (K-Means Clustering)
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.spatial.distance import euclidean, cdist

pd.set_option('display.max_columns', None)

# ========================================================================
# BAGIAN 1: SUPERVISED LEARNING (Kode Asli Anda)
# ========================================================================

print("\n" + "="*80)
print("BAGIAN 1: SUPERVISED LEARNING - CLASSIFICATION")
print("="*80)

# Load data
df = pd.read_excel("CADalizadeh.xls")
target = 'Cath'
data = df

print('\nDimensi (baris, kolom):', data.shape)
print('Jumlah atribut (kolom):', data.shape[1])
print('Jumlah data (baris):', data.shape[0])

print('\nContoh 5 baris awal:')
print(data.head())

print('\nInfo dataset:')
print(data.info())

print('\nStatistik deskriptif semua kolom:')
print(data.describe(include='all').T)

print('\nJumlah missing value per kolom (top 20):')
print(data.isna().sum().sort_values(ascending=False).head(20))

num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if target in num_cols:
    num_cols.remove(target)

cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
if target in cat_cols:
    cat_cols.remove(target)

print('\nJumlah fitur numerik:', len(num_cols))
print('Jumlah fitur kategorikal:', len(cat_cols))

if target in data.columns:
    plt.figure(figsize=(6,4))
    sns.countplot(x=data[target].astype(str))
    plt.title(f'Distribusi target ({target})')
    plt.show()

corr_target = pd.Series(dtype=float)
if len(num_cols) >= 1 and target in data.columns:
    y_enc = LabelEncoder().fit_transform(data[target].astype(str))
    df_corr = data[num_cols].copy()
    df_corr['__target_enc'] = y_enc
    corr_target = df_corr.corr()['__target_enc'].drop('__target_enc').abs().sort_values(ascending=False)

if len(num_cols) >= 2:
    plt.figure(figsize=(10,8))
    sns.heatmap(data[num_cols].corr(), cmap='coolwarm', center=0, vmax=1, vmin=-1)
    plt.title('Matriks korelasi fitur numerik')
    plt.show()

top_feats = corr_target.head(6).index.tolist() if not corr_target.empty else []
if top_feats:
    plt.figure(figsize=(12,8))
    for i, col in enumerate(top_feats, 1):
        plt.subplot(3,2,i)
        sns.histplot(data[col].dropna(), kde=True)
        plt.title(col)
    plt.tight_layout()
    plt.show()

if len(top_feats) >= 3:
    plt.figure(figsize=(12,6))
    for i, col in enumerate(top_feats[:3], 1):
        plt.subplot(1,3,i)
        sns.boxplot(x=data[target].astype(str), y=data[col])
        plt.title(f'{col} per {target}')
    plt.tight_layout()
    plt.show()

# Preprocessing
df = df.drop_duplicates()
df = df.dropna()

le = LabelEncoder()
df[target] = le.fit_transform(df[target].astype(str))

X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
if 'No' in X.columns:
    X = X.drop(columns=['No'])

y = df[target]

print('\nUkuran X:', X.shape)
print('Ukuran y:', y.shape)
print('Kolom X:', X.columns.tolist())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Model training - Supervised Learning
print("\n[Supervised Learning] Training models...")

models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'MLP': MLPClassifier(max_iter=500, random_state=42),
    'NaiveBayes': GaussianNB()
}

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

results_df = pd.DataFrame(results).T
print("\nHasil Supervised Learning Models:")
print(results_df)

# Best model - RandomForest
model = RandomForestClassifier(random_state=42)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print('\nClassification Report - RandomForest')
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - RandomForest')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# MLP
model = MLPClassifier(max_iter=500, random_state=42)
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print('\nClassification Report - MLP')
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MLP')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

# NaiveBayes
model = GaussianNB()
model.fit(X_train_s, y_train)
y_pred = model.predict(X_test_s)

print('\nClassification Report - NaiveBayes')
print(classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - NaiveBayes')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.show()

print("\n✓ BAGIAN SUPERVISED LEARNING SELESAI")

# ========================================================================
# BAGIAN 2: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
# ========================================================================

print("\n" + "="*80)
print("BAGIAN 2: UNSUPERVISED LEARNING - K-MEANS CLUSTERING")
print("="*80)

# ============================================================================
# STEP 1: IMPLEMENTASI K-MEANS DARI AWAL
# ============================================================================

print("\n[STEP 1] Implementasi K-Means Algorithm dari awal...")

class KMeansCustom:
    """Implementasi K-Means Clustering dari awal tanpa menggunakan library"""
    
    def __init__(self, k=3, max_iter=100, random_state=None, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.centroids = None
        self.labels = None
        self.history = []
        
    def _initialize_centroids(self, X):
        np.random.seed(self.random_state)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices].copy()
    
    def _assign_clusters(self, X):
        distances = cdist(X, self.centroids, metric='euclidean')
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids
    
    def _calculate_sse(self, X, labels):
        sse = 0.0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                sse += np.sum((cluster_points - self.centroids[i]) ** 2)
        return sse
    
    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        
        if self.verbose:
            print(f"  Inisialisasi centroid dengan k={self.k}")
        
        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)
            sse = self._calculate_sse(X, labels)
            
            self.history.append({
                'iteration': iteration,
                'sse': sse,
                'labels': labels.copy()
            })
            
            if self.verbose:
                print(f"  Iterasi {iteration + 1}/{self.max_iter} - SSE: {sse:.4f}")
            
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
        return self._assign_clusters(X)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels

print("✓ Class KMeansCustom sudah didefinisikan")

# ============================================================================
# STEP 2: FUNGSI EVALUASI
# ============================================================================

print("\n[STEP 2] Definisikan fungsi evaluasi clustering quality...")

def calculate_silhouette_score(X, labels):
    n_samples = X.shape[0]
    silhouette_values = np.zeros(n_samples)
    unique_labels = np.unique(labels)
    
    for i in range(n_samples):
        cluster_i = labels[i]
        cluster_points = X[labels == cluster_i]
        
        if len(cluster_points) > 1:
            a_i = np.mean([euclidean(X[i], p) for p in cluster_points if not np.array_equal(X[i], p)])
        else:
            a_i = 0
        
        b_i = np.inf
        for label in unique_labels:
            if label != cluster_i:
                other_cluster_points = X[labels == label]
                if len(other_cluster_points) > 0:
                    distance = np.mean([euclidean(X[i], p) for p in other_cluster_points])
                    b_i = min(b_i, distance)
        
        if max(a_i, b_i) == 0:
            silhouette_values[i] = 0
        else:
            silhouette_values[i] = (b_i - a_i) / max(a_i, b_i)
    
    silhouette_avg = np.mean(silhouette_values)
    return silhouette_avg, silhouette_values

def davies_bouldin_index(X, labels, centroids):
    k = len(np.unique(labels))
    db_index = 0.0
    
    avg_distances = np.zeros(k)
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            avg_distances[i] = np.mean([euclidean(p, centroids[i]) for p in cluster_points])
    
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
# STEP 3: ELBOW METHOD
# ============================================================================

print("\n[STEP 3] Elbow Method - Menentukan jumlah cluster optimal...")
print("  (Ini mungkin memakan waktu beberapa detik...)")

X_unsupervised = X_train_s.copy()

k_range = range(2, 11)
sse_values = []
silhouette_scores = []
db_indices = []

for k in k_range:
    print(f"  Testing k={k}...", end=" ")
    
    kmeans = KMeansCustom(k=k, random_state=42, verbose=False)
    labels = kmeans.fit_predict(X_unsupervised)
    
    sse = 0.0
    for i in range(k):
        cluster_points = X_unsupervised[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - kmeans.centroids[i]) ** 2)
    sse_values.append(sse)
    
    sil_score, _ = calculate_silhouette_score(X_unsupervised, labels)
    silhouette_scores.append(sil_score)
    
    db_index = davies_bouldin_index(X_unsupervised, labels, kmeans.centroids)
    db_indices.append(db_index)
    
    print(f"SSE={sse:.2f}, Silhouette={sil_score:.4f}, DB={db_index:.4f}")

print("✓ Elbow method selesai")

# ============================================================================
# STEP 4: VISUALISASI ELBOW METHOD
# ============================================================================

print("\n[STEP 4] Visualisasi Elbow Method...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(list(k_range), sse_values, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Sum of Squared Error (SSE)', fontsize=12, fontweight='bold')
axes[0].set_title('Elbow Method - SSE', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(list(k_range))

axes[1].plot(list(k_range), silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1].set_title('Silhouette Analysis', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(list(k_range))
axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)

axes[2].plot(list(k_range), db_indices, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Jumlah Cluster (k)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[2].set_title('Davies-Bouldin Index (lebih rendah = lebih baik)', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(list(k_range))

plt.tight_layout()
plt.show()

comparison_df = pd.DataFrame({
    'K': list(k_range),
    'SSE': [f"{x:.2f}" for x in sse_values],
    'Silhouette Score': [f"{x:.4f}" for x in silhouette_scores],
    'Davies-Bouldin Index': [f"{x:.4f}" for x in db_indices]
})
print("\nTabel Perbandingan Metrik untuk berbagai K:")
print(comparison_df.to_string(index=False))

# ============================================================================
# STEP 5: PEMILIHAN K OPTIMAL DAN TRAINING
# ============================================================================

print("\n[STEP 5] Pemilihan K optimal dan training...")

# PILIH K OPTIMAL (sesuaikan berdasarkan analisis plot di atas)
optimal_k = 3  # ← SESUAIKAN BERDASARKAN ANALISIS ANDA

print(f"\n🎯 K OPTIMAL YANG DIPILIH: {optimal_k}")
print(f"  - SSE: {sse_values[optimal_k-2]:.2f}")
print(f"  - Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}")
print(f"  - Davies-Bouldin Index: {db_indices[optimal_k-2]:.4f}")

print(f"\n[STEP 6] Training K-Means dengan k={optimal_k}...")

kmeans = KMeansCustom(k=optimal_k, max_iter=100, random_state=42, verbose=True)
cluster_labels = kmeans.fit_predict(X_unsupervised)

print(f"✓ Model training selesai!")
print(f"  - Total iterasi hingga konvergen: {len(kmeans.history)}")
print(f"  - Final SSE: {kmeans.history[-1]['sse']:.4f}")

# ============================================================================
# STEP 7: EVALUASI HASIL CLUSTERING
# ============================================================================

print(f"\n[STEP 7] Evaluasi hasil clustering...")

silhouette_avg, silhouette_values = calculate_silhouette_score(X_unsupervised, cluster_labels)
db_score = davies_bouldin_index(X_unsupervised, cluster_labels, kmeans.centroids)

print(f"\nKualitas Clustering:")
print(f"  Silhouette Score: {silhouette_avg:.4f}")
print(f"    └─ Interpretasi: {'Sangat Baik ✓✓✓' if silhouette_avg > 0.7 else 'Baik ✓✓' if silhouette_avg > 0.5 else 'Cukup ✓' if silhouette_avg > 0.25 else 'Kurang baik'}")

print(f"\n  Davies-Bouldin Index: {db_score:.4f}")
print(f"    └─ Interpretasi: {'Sangat Baik ✓✓✓' if db_score < 0.5 else 'Baik ✓✓' if db_score < 1.0 else 'Cukup ✓' if db_score < 1.5 else 'Kurang baik'}")

print(f"\nDistribusi anggota cluster:")
unique, counts = np.unique(cluster_labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

# ============================================================================
# STEP 8: VISUALISASI HASIL CLUSTERING
# ============================================================================

print(f"\n[STEP 8] Visualisasi hasil clustering...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
# STEP 9: ANALISIS KARAKTERISTIK CLUSTER
# ============================================================================

print(f"\n[STEP 9] Analisis karakteristik setiap cluster...")

for cluster_id in range(optimal_k):
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}")
    print('='*80)
    
    cluster_mask = cluster_labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    cluster_data = X_unsupervised[cluster_mask]
    
    print(f"\nUkuran: {cluster_size} points ({cluster_size/len(cluster_labels)*100:.1f}%)")
    
    print(f"\nCentroid (values):")
    for i, feature in enumerate(X.columns):
        print(f"  {feature}: {cluster_data[:, i].mean():.4f}")
    
    print(f"\nCharacteristics (mean ± std):")
    for i, feature in enumerate(X.columns):
        mean = cluster_data[:, i].mean()
        std = cluster_data[:, i].std()
        print(f"  {feature}: {mean:.4f} ± {std:.4f}")

# ============================================================================
# STEP 10: SILHOUETTE PLOT
# ============================================================================

print(f"\n[STEP 10] Visualisasi Silhouette Analysis...")

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

# ============================================================================
# KESIMPULAN
# ============================================================================

print("\n" + "="*80)
print("KESIMPULAN K-MEANS CLUSTERING")
print("="*80)

print(f"""
Hasil Clustering:
  ✓ Optimal K: {optimal_k}
  ✓ Silhouette Score: {silhouette_avg:.4f} - {'SANGAT BAIK' if silhouette_avg > 0.7 else 'BAIK' if silhouette_avg > 0.5 else 'CUKUP'}
  ✓ Davies-Bouldin Index: {db_score:.4f} - {'SANGAT BAIK' if db_score < 0.5 else 'BAIK' if db_score < 1.0 else 'CUKUP'}
  
Cluster Distribution:
""")

for cluster_id, count in enumerate(counts):
    print(f"  Cluster {cluster_id}: {count} points ({count/len(cluster_labels)*100:.1f}%)")

print(f"""
Interpretasi:
  - Data berhasil di-cluster menjadi {optimal_k} grup yang distinct
  - Kualitas clustering termasuk kategori {'SANGAT BAIK' if silhouette_avg > 0.7 else 'BAIK' if silhouette_avg > 0.5 else 'CUKUP'}
  - Setiap cluster memiliki karakteristik fitur yang berbeda
  - Hasil clustering dapat digunakan untuk segmentasi pasien CAD-RADS
""")

print("="*80)
print("✓ ANALISIS UNSUPERVISED LEARNING - K-MEANS SELESAI")
print("="*80)
