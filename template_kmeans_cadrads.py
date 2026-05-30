"""
TEMPLATE: UNSUPERVISED LEARNING - K-MEANS CLUSTERING
Dataset: CAD-RADS
Mata Kuliah: Pembelajaran Mesin
"""

# ============================================================================
# SETUP - Jalankan di Google Colab
# ============================================================================

# Jika menggunakan Google Colab, uncomment baris di bawah untuk mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean, cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DAN EKSPLORASI DATA
# ============================================================================

print("="*70)
print("1. LOAD DAN EKSPLORASI DATA")
print("="*70)

# Load dataset CAD-RADS
# Ganti path sesuai dengan lokasi file Anda
df = pd.read_csv('data_cad_rads.csv')  # Ganti dengan path yang sesuai

print("\nInfo Dataset:")
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

# ============================================================================
# 2. PREPROCESSING DATA
# ============================================================================

print("\n" + "="*70)
print("2. PREPROCESSING DATA")
print("="*70)

# PILIHAN 1: Hapus baris dengan missing values
df_clean = df.dropna()

# PILIHAN 2: Isi missing values dengan mean (uncomment jika digunakan)
# df_clean = df.fillna(df.mean())

print(f"\nShape setelah cleaning: {df_clean.shape}")

# Pilih atribut yang akan digunakan
# PENTING: Jangan gunakan kolom 'Cath' (itu untuk supervised learning)
# Sesuaikan dengan nama kolom di dataset Anda

# CONTOH (sesuaikan dengan dataset Anda):
selected_features = [col for col in df_clean.columns if col != 'Cath' and col != 'ID']
# Atau pilih secara spesifik:
# selected_features = ['Age', 'BMI', 'Diabetes', 'Hypertension', 'ChestPain']

print(f"\nFitur yang digunakan ({len(selected_features)} fitur):")
print(selected_features)

X = df_clean[selected_features].values

print(f"\nShape data untuk clustering: {X.shape}")
print(f"Statistika data sebelum normalisasi:")
print(f"  Mean: {X.mean(axis=0)}")
print(f"  Std: {X.std(axis=0)}")

# Normalisasi/Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nSetelah normalisasi:")
print(f"  Mean: {X_scaled.mean(axis=0):.6f}")
print(f"  Std: {X_scaled.std(axis=0):.6f}")

# ============================================================================
# 3. PASTE KODE K-MEANS DARI SINI
# ============================================================================
# Copy-paste seluruh kode dari file kmeans_implementation.py
# Mulai dari class KMeansCustom sampai semua fungsi evaluasi

# [COPY PASTE KODE K-MEANS DI SINI]

# ============================================================================
# 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL
# ============================================================================

print("\n" + "="*70)
print("4. MENENTUKAN JUMLAH CLUSTER OPTIMAL - ELBOW METHOD")
print("="*70)

# Jalankan elbow method
sse_values, silhouette_scores, db_indices = elbow_method(
    X_scaled, 
    k_range=range(2, 11)
)

# Visualisasi
plot_elbow_curve(range(2, 11), sse_values, silhouette_scores, db_indices)

# Analisis hasil
print("\nAnalisis Elbow Method:")
print("-"*70)
print("\nBerdasarkan grafik di atas, tentukan k optimal dengan mempertimbangkan:")
print("1. SSE: Cari 'elbow' - titik di mana SSE mulai stabil")
print("2. Silhouette Score: Semakin tinggi semakin baik (max 1)")
print("3. Davies-Bouldin Index: Semakin rendah semakin baik")

# PILIH K OPTIMAL (sesuaikan berdasarkan analisis Anda)
optimal_k = 3  # UBAH SESUAI DENGAN ANALISIS ANDA

print(f"\n✓ K optimal yang dipilih: {optimal_k}")
print(f"  SSE: {sse_values[optimal_k-2]:.4f}")
print(f"  Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}")
print(f"  Davies-Bouldin Index: {db_indices[optimal_k-2]:.4f}")

# ============================================================================
# 5. TRAINING K-MEANS DENGAN K OPTIMAL
# ============================================================================

print("\n" + "="*70)
print("5. TRAINING MODEL K-MEANS")
print("="*70)

kmeans = KMeansCustom(k=optimal_k, max_iter=100, random_state=42, verbose=True)
labels = kmeans.fit_predict(X_scaled)

print(f"\n✓ Model trained successfully!")
print(f"  Jumlah iterasi hingga konvergen: {len(kmeans.history)}")
print(f"  Final SSE: {kmeans.history[-1]['sse']:.4f}")

# ============================================================================
# 6. ANALISIS HASIL CLUSTERING
# ============================================================================

print("\n" + "="*70)
print("6. ANALISIS HASIL CLUSTERING")
print("="*70)

# Hitung metrik kualitas
silhouette_avg, silhouette_values = calculate_silhouette_score(X_scaled, labels)
db_score = davies_bouldin_index(X_scaled, labels, kmeans.centroids)

print(f"\nKualitas Clustering:")
print(f"  Silhouette Score: {silhouette_avg:.4f}")
print(f"    └─ Interpretasi: {'Sangat Baik' if silhouette_avg > 0.7 else 'Baik' if silhouette_avg > 0.5 else 'Cukup' if silhouette_avg > 0.25 else 'Kurang baik'}")
print(f"  Davies-Bouldin Index: {db_score:.4f}")
print(f"    └─ Interpretasi: {'Sangat Baik' if db_score < 0.5 else 'Baik' if db_score < 1 else 'Cukup'}")

# Distribusi cluster
print(f"\nDistribusi anggota cluster:")
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(labels)) * 100
    print(f"  Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

# ============================================================================
# 7. VISUALISASI HASIL
# ============================================================================

print("\n" + "="*70)
print("7. VISUALISASI HASIL CLUSTERING")
print("="*70)

# Plot clustering (menggunakan 2 fitur pertama)
plot_clusters_2d(X_scaled, labels, kmeans.centroids, 
                title=f"K-Means Clustering dengan k={optimal_k}")

# Silhouette Analysis
plot_silhouette_analysis(X_scaled, labels, optimal_k)

# ============================================================================
# 8. INTERPRETASI DAN ANALISIS FITUR DALAM SETIAP CLUSTER
# ============================================================================

print("\n" + "="*70)
print("8. KARAKTERISTIK SETIAP CLUSTER")
print("="*70)

# Denormalisasi centroid untuk interpretasi
centroids_original = scaler.inverse_transform(kmeans.centroids)

for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id}:")
    print("-"*70)
    
    cluster_mask = labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    
    print(f"  Ukuran: {cluster_size} points")
    print(f"  Centroid (nilai original):")
    for i, feature in enumerate(selected_features):
        print(f"    {feature}: {centroids_original[cluster_id, i]:.4f}")
    
    # Karakteristik cluster
    cluster_data = X[cluster_mask]
    print(f"  Karakteristik:")
    for i, feature in enumerate(selected_features):
        mean_val = cluster_data[:, i].mean()
        print(f"    {feature}: mean={mean_val:.4f}, std={cluster_data[:, i].std():.4f}")

# ============================================================================
# 9. SIMPAN HASIL
# ============================================================================

print("\n" + "="*70)
print("9. SIMPAN HASIL CLUSTERING")
print("="*70)

# Tambahkan cluster label ke dataframe original
df_clean['Cluster'] = labels

# Simpan hasil
# output_path = '/content/drive/MyDrive/hasil_clustering.csv'  # Untuk Colab
output_path = 'hasil_clustering.csv'
df_clean.to_csv(output_path, index=False)

print(f"✓ Hasil clustering disimpan ke: {output_path}")

# Simpan summary
summary = pd.DataFrame({
    'Metric': ['Optimal K', 'Silhouette Score', 'Davies-Bouldin Index', 'Total Samples', 'Total Features'],
    'Value': [optimal_k, f'{silhouette_avg:.4f}', f'{db_score:.4f}', len(X_scaled), len(selected_features)]
})

print("\nRingkasan Hasil:")
print(summary.to_string(index=False))

print("\n" + "="*70)
print("✓ PROSES CLUSTERING SELESAI")
print("="*70)

# ============================================================================
# TUGAS LANJUTAN (OPSIONAL - untuk nilai lebih)
# ============================================================================

print("\n" + "="*70)
print("TUGAS LANJUTAN (Opsional)")
print("="*70)

print("""
Untuk meningkatkan analisis, Anda bisa mencoba:

1. MULTIPLE INITIALIZATION
   - Jalankan K-Means berkali-kali dengan seed berbeda
   - Bandingkan hasilnya
   - Pilih yang memiliki SSE terkecil

2. HANDLING OUTLIERS
   - Identifikasi outliers dengan Isolation Forest atau Z-score
   - Coba remove outliers dan re-run clustering
   - Bandingkan hasilnya

3. FEATURE SELECTION
   - Coba kombinasi fitur berbeda
   - Lihat bagaimana perubahan clustering quality
   - Analisis mana fitur yang paling berpengaruh

4. COMPARISON DENGAN ALGORITMA LAIN
   - Bandingkan hasil dengan Hierarchical Clustering
   - Bandingkan dengan Gaussian Mixture Model (GMM)
   - Gunakan library sklearn untuk perbandingan (tapi jangan untuk implementasi K-Means)

5. CLUSTER VALIDATION
   - Cross-validation dengan berbagai k
   - Bootstrap resampling untuk stability assessment
   - Silhouette analysis untuk setiap sample
""")
