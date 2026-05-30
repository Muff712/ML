# PANDUAN LENGKAP K-MEANS CLUSTERING DARI AWAL
## Tugas Besar Pembelajaran Mesin - CAD-RADS Dataset

---

## 📋 DAFTAR ISI

1. [Pengenalan K-Means](#pengenalan-kmeans)
2. [Algoritma K-Means](#algoritma-kmeans)
3. [Implementasi dari Awal](#implementasi-dari-awal)
4. [Metrik Evaluasi](#metrik-evaluasi)
5. [Cara Penggunaan](#cara-penggunaan)
6. [Troubleshooting](#troubleshooting)

---

## 🎯 Pengenalan K-Means

### Apa itu K-Means?

K-Means adalah algoritma clustering **unsupervised learning** yang membagi data menjadi K kluster. Setiap kluster direpresentasikan oleh satu **centroid** (titik pusat), dan setiap data point diassign ke centroid terdekat.

### Karakteristik K-Means

| Aspek | Deskripsi |
|-------|-----------|
| **Tipe Clustering** | Partitional (Centroid-based) |
| **Hard/Soft** | Hard Clustering (setiap point hanya dalam 1 cluster) |
| **Monothetic/Polythetic** | Polythetic (menggunakan semua fitur) |
| **Kompleksitas** | O(n × k × i × d) - n: samples, k: clusters, i: iterations, d: dimensions |
| **Scalability** | Baik untuk dataset besar |
| **Parameter** | K (jumlah cluster) |

### Kapan Menggunakan K-Means?

✅ **Cocok untuk:**
- Data dengan bentuk cluster yang rounded/convex
- Dataset besar
- Ketika Anda sudah tahu approximate number of clusters
- Ketika Anda perlu algoritma yang cepat

❌ **Tidak cocok untuk:**
- Cluster dengan bentuk kompleks/non-convex
- Cluster dengan ukuran sangat berbeda
- Ketika ada banyak outliers
- Ketika tidak tahu jumlah cluster

---

## 🔧 Algoritma K-Means

### Langkah-Langkah Algoritma

```
PSEUDOCODE K-MEANS:
─────────────────────────────────────
1. INISIALISASI
   - Pilih K titik random dari data sebagai centroid awal
   
2. LOOP ITERASI (sampai konvergen atau max_iter tercapai):
   
   a. ASSIGN PHASE
      - Untuk setiap data point:
        * Hitung jarak ke semua centroid
        * Assign ke centroid terdekat
      
   b. UPDATE PHASE
      - Untuk setiap cluster:
        * Hitung mean dari semua points dalam cluster
        * Update centroid ke mean tersebut
   
   c. CHECK CONVERGENCE
      - Jika semua centroid tidak berubah → STOP
      - Else → ulangi ke step a

3. RETURN
   - Centroid final
   - Label cluster untuk setiap data point
─────────────────────────────────────
```

### Visual Step-by-Step

```
ITERASI 1 (Inisialisasi dengan centroid random):
┌─────────────────────┐
│ •      C1           │
│    • •              │
│      •              │
│           • C2 •    │
│            • •      │
│        • •          │
└─────────────────────┘
C1, C2 = centroid (dipilih random dari data)

ITERASI 2 (Assign points ke centroid terdekat):
┌─────────────────────┐
│ ●●●     C1          │
│ ●●● ●               │
│  ● ●                │
│      ●●● C2 ●●●●    │
│      ●●●● ●●●       │
│        ●●●●         │
└─────────────────────┘
● = points di cluster 1, ● = points di cluster 2

ITERASI 3 (Update centroid sebagai mean):
┌─────────────────────┐
│ ●●●   C1'●          │
│ ●●● ●               │
│  ● ●                │
│      ●●● ●C2' ●●●● │
│      ●●●● ●●●       │
│        ●●●●         │
└─────────────────────┘
C1', C2' = centroid yang sudah di-update

... (ulangi sampai centroid tidak berubah)
```

### Jarak Euclidean

Formula yang digunakan untuk menghitung jarak:

```
distance = sqrt((x1-x2)² + (y1-y2)² + ... + (n1-n2)²)

Contoh 2D:
Point A = (2, 3)
Point B = (5, 7)
distance = sqrt((2-5)² + (3-7)²) = sqrt(9 + 16) = sqrt(25) = 5
```

### Update Centroid

Centroid baru dihitung sebagai **mean (rata-rata)** dari semua points dalam cluster:

```
centroid_baru = (sum semua points dalam cluster) / (jumlah points)

Contoh:
Cluster 1 memiliki points: [1,1], [2,2], [3,3]
centroid_baru = ([1+2+3]/3, [1+2+3]/3) = (2, 2)
```

---

## 💻 Implementasi dari Awal

### File-File yang Disediakan

1. **`kmeans_implementation.py`**
   - Implementasi lengkap kelas `KMeansCustom`
   - Fungsi-fungsi evaluasi clustering quality
   - Fungsi visualisasi
   - Contoh penggunaan dengan data dummy

2. **`template_kmeans_cadrads.py`**
   - Template siap pakai untuk dataset CAD-RADS
   - Workflow lengkap: load → preprocess → train → evaluate → visualize
   - Sudah terstruktur dengan baik

### Penjelasan Implementasi

#### Class KMeansCustom

```python
class KMeansCustom:
    def __init__(self, k=3, max_iter=100, random_state=None, verbose=False):
        """
        k: jumlah cluster
        max_iter: iterasi maksimal
        random_state: seed untuk reproducibility
        verbose: print informasi iterasi
        """
```

#### Method Utama

**1. `_initialize_centroids(X)`**
```python
# Pilih K random points dari data sebagai centroid awal
random_indices = np.random.choice(X.shape[0], self.k, replace=False)
return X[random_indices].copy()
```

**2. `_assign_clusters(X)`**
```python
# Hitung jarak Euclidean ke semua centroid
distances = cdist(X, self.centroids, metric='euclidean')  # shape: (n_samples, k)

# Ambil index centroid terdekat untuk setiap point
labels = np.argmin(distances, axis=1)  # shape: (n_samples,)
return labels
```

**3. `_update_centroids(X, labels)`**
```python
# Update centroid sebagai mean dari setiap cluster
for i in range(self.k):
    cluster_points = X[labels == i]  # Ambil semua points di cluster i
    if len(cluster_points) > 0:
        new_centroids[i] = cluster_points.mean(axis=0)  # Hitung mean
    else:
        new_centroids[i] = self.centroids[i]  # Keep yang lama jika cluster kosong
return new_centroids
```

**4. `fit(X)`**
```python
# Main training loop
for iteration in range(self.max_iter):
    # Step 1: Assign
    labels = self._assign_clusters(X)
    
    # Step 2: Update
    new_centroids = self._update_centroids(X, labels)
    
    # Step 3: Check convergence
    if np.allclose(self.centroids, new_centroids):
        break  # Sudah konvergen
    
    self.centroids = new_centroids
return self
```

---

## 📊 Metrik Evaluasi

Bagian paling penting: **Bagaimana kita tahu hasil clustering bagus atau tidak?**

### 1. Sum of Squared Error (SSE)

**Konsep:** Jumlah jarak kuadrat setiap point ke centroid-nya.

```
SSE = Σ (point - centroid)²

Semakin kecil SSE → semakin baik clustering
```

**Implementasi:**
```python
def _calculate_sse(self, X, labels):
    sse = 0.0
    for i in range(self.k):
        cluster_points = X[labels == i]
        if len(cluster_points) > 0:
            sse += np.sum((cluster_points - self.centroids[i]) ** 2)
    return sse
```

**Interpretasi:**
- SSE selalu menurun seiring bertambahnya k
- Gunakan **Elbow Method** untuk menemukan "titik siku" (elbow point)

### 2. Silhouette Score

**Konsep:** Mengukur seberapa baik setiap point cocok dengan clusternya dibanding dengan cluster tetangga.

```
silhouette = (b - a) / max(a, b)

Dimana:
- a = rata-rata jarak point ke semua points dalam cluster yang sama
- b = rata-rata jarak point ke points dalam cluster terdekat

Range: -1 hingga 1
- Nilai dekat 1: point sangat cocok dengan clusternya
- Nilai dekat 0: point di antara dua cluster
- Nilai negatif: point mungkin masuk cluster yang salah
```

**Interpretasi:**
| Score | Interpretasi |
|-------|--------------|
| 0.7 - 1.0 | Struktur clustering sangat baik |
| 0.5 - 0.7 | Struktur clustering baik |
| 0.25 - 0.5 | Struktur clustering cukup |
| < 0.25 | Struktur clustering lemah |

**Keuntungan:**
- Mempertimbangkan separation (jarak ke cluster lain)
- Tidak bergantung pada jumlah cluster
- Bisa dihitung untuk setiap sample (bukan aggregate)

### 3. Davies-Bouldin Index (DB Index)

**Konsep:** Mengukur rasio average intra-cluster distance terhadap inter-cluster distance.

```
DB Index = (1/k) Σ max(D_i,j)

Dimana:
D_i,j = (avg_dist_cluster_i + avg_dist_cluster_j) / distance(centroid_i, centroid_j)

Semakin rendah DB Index → semakin baik clustering
```

**Interpretasi:**
| DB Index | Interpretasi |
|----------|--------------|
| < 0.5 | Clustering sangat baik |
| 0.5 - 1.0 | Clustering baik |
| > 1.0 | Clustering kurang baik |

---

## 🚀 Cara Penggunaan

### Setup Awal

```python
# Import
from kmeans_implementation import *
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('data_cad_rads.csv')

# Pilih fitur (jangan gunakan 'Cath')
features = ['Age', 'BMI', 'Diabetes', 'Hypertension']  # Sesuaikan
X = df[features].values

# Normalisasi (SANGAT PENTING!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Step 1: Tentukan K Optimal dengan Elbow Method

```python
# Coba berbagai nilai k
sse_values, silhouette_scores, db_indices = elbow_method(
    X_scaled, 
    k_range=range(2, 11)
)

# Visualisasi
plot_elbow_curve(range(2, 11), sse_values, silhouette_scores, db_indices)

# Analisis dan pilih k
# Lihat grafik, cari elbow point di SSE
# Lihat Silhouette score yang tinggi
# Lihat DB Index yang rendah
```

### Step 2: Training dengan K Optimal

```python
# Pilih k berdasarkan analisis
optimal_k = 3  # Contoh

# Create dan train model
kmeans = KMeansCustom(k=optimal_k, max_iter=100, random_state=42, verbose=True)
labels = kmeans.fit_predict(X_scaled)
```

### Step 3: Evaluasi Hasil

```python
# Hitung metrik
silhouette_avg, _ = calculate_silhouette_score(X_scaled, labels)
db_index = davies_bouldin_index(X_scaled, labels, kmeans.centroids)

print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {db_index:.4f}")

# Cek distribusi cluster
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    print(f"Cluster {cluster_id}: {count} points ({count/len(labels)*100:.1f}%)")
```

### Step 4: Analisis Karakteristik Cluster

```python
# Denormalisasi centroid
centroids_original = scaler.inverse_transform(kmeans.centroids)

# Analisis setiap cluster
for cluster_id in range(optimal_k):
    print(f"\nCluster {cluster_id}:")
    cluster_mask = labels == cluster_id
    cluster_data = X[cluster_mask]
    
    # Tampilkan karakteristik
    for i, feature in enumerate(features):
        mean_val = cluster_data[:, i].mean()
        print(f"  {feature}: mean={mean_val:.4f}")
```

### Step 5: Visualisasi

```python
# Plot clustering
plot_clusters_2d(X_scaled, labels, kmeans.centroids, 
                title=f"K-Means (k={optimal_k})")

# Silhouette analysis
plot_silhouette_analysis(X_scaled, labels, optimal_k)
```

---

## 🔍 Troubleshooting

### Problem 1: Convergence Lambat

**Gejala:** Model tidak konvergen dalam 100 iterasi

**Solusi:**
```python
# Increase max_iter
kmeans = KMeansCustom(k=optimal_k, max_iter=300, verbose=True)

# Atau sesuaikan learning rate implisit dengan initial seed yang lebih baik
kmeans = KMeansCustom(k=optimal_k, random_state=123, max_iter=100)
```

### Problem 2: Hasil Berbeda-Beda di Setiap Run

**Gejala:** Hasil clustering berubah setiap kali dijalankan

**Penyebab:** Inisialisasi centroid random
**Solusi:**
```python
# Gunakan random_state yang tetap
kmeans = KMeansCustom(k=optimal_k, random_state=42)

# Atau jalankan berkali-kali dan pilih hasil terbaik
best_sse = float('inf')
best_model = None

for seed in range(10):
    kmeans = KMeansCustom(k=optimal_k, random_state=seed)
    kmeans.fit(X_scaled)
    sse = kmeans.history[-1]['sse']
    
    if sse < best_sse:
        best_sse = sse
        best_model = kmeans

# Gunakan best_model
labels = best_model.labels
```

### Problem 3: Silhouette Score Negatif

**Gejala:** Silhouette score < 0

**Penyebab:** 
- K terlalu besar atau terlalu kecil
- Data tidak memiliki struktur cluster yang jelas
- Ada outliers

**Solusi:**
```python
# Coba k yang berbeda
for k in [2, 3, 4, 5]:
    kmeans = KMeansCustom(k=k)
    labels = kmeans.fit_predict(X_scaled)
    sil_score, _ = calculate_silhouette_score(X_scaled, labels)
    print(f"k={k}: Silhouette={sil_score:.4f}")

# Atau remove outliers
from sklearn.preprocessing import RobustScaler
# Gunakan RobustScaler untuk kurangi pengaruh outliers
```

### Problem 4: Cluster Kosong

**Gejala:** Ada cluster yang tidak memiliki anggota

**Penyebab:** K terlalu besar, atau centroid awal kurang baik

**Solusi:**
```python
# Centroid akan keep nilai lama jika cluster kosong (sudah di-handle di code)
# Coba kurangi k atau ganti random_state

# Atau gunakan k-means++
# (Inisialisasi lebih pintar, tapi harus implement sendiri)
```

---

## 📝 Checklist untuk Laporan

Pastikan laporan Anda mencakup:

### 1. Pendahuluan
- [ ] Definisi K-Means
- [ ] Karakteristik (hard/soft, partitional, dll)
- [ ] Kelebihan dan kekurangan K-Means
- [ ] Mengapa memilih K-Means untuk dataset ini

### 2. Pengolahan Data
- [ ] Deskripsi dataset (jumlah sample, fitur)
- [ ] Fitur yang digunakan dan alasan pemilihannya
- [ ] Preprocessing steps (handling missing values, normalisasi)
- [ ] Statistika deskriptif data

### 3. Metodologi
- [ ] Penjelasan algoritma K-Means step-by-step
- [ ] Metrik evaluasi yang digunakan
- [ ] Cara menentukan K optimal

### 4. Hasil Eksperimen
- [ ] Elbow method plot
- [ ] Silhouette analysis plot
- [ ] Hasil clustering dengan K optimal
- [ ] Nilai-nilai metrik evaluasi
- [ ] Karakteristik setiap cluster

### 5. Analisis dan Diskusi
- [ ] Interpretasi hasil clustering
- [ ] Perbedaan antar cluster
- [ ] Apakah clustering sesuai dengan domain knowledge?
- [ ] Keterbatasan hasil

### 6. Kesimpulan
- [ ] Ringkasan temuan utama
- [ ] Implikasi hasil
- [ ] Saran untuk pekerjaan di masa depan

---

## 📚 Referensi

1. MacQueen, J. (1967). "Some Methods for classification and Analysis of Multivariate Observations"
2. Jain, A. K. (2010). "Data clustering: 50 years beyond K-means"
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
4. Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"

---

## 🆘 Bantuan Cepat

**Q: Bagaimana cara memilih K?**
A: Gunakan Elbow Method dan Silhouette Analysis. Cari k yang SSE-nya mulai stabil dan Silhouette score-nya tinggi.

**Q: Harus normalisasi data?**
A: Ya! K-Means sangat sensitif terhadap scale. Gunakan StandardScaler.

**Q: Boleh pakai library?**
A: **Tidak** untuk implementasi K-Means. Tapi boleh pakai untuk preprocessing, visualisasi, dan evaluasi.

**Q: Berapa iterasi yang pas?**
A: Biasanya 100 iterasi sudah cukup. Model akan converge lebih cepat.

**Q: Bagaimana jika ada NaN value?**
A: Hapus baris dengan NaN atau isi dengan mean. Jangan lupa dokumentasikan di laporan.

---

**Semoga sukses dengan tugas besar Anda!** 🎓

Jika ada pertanyaan, silakan tanya di kelas atau di forum diskusi.
