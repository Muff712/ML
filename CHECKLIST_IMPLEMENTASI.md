# QUICK REFERENCE: K-MEANS IMPLEMENTATION CHECKLIST
## Unsupervised Learning - Tugas Besar Pembelajaran Mesin

---

## 📋 PRE-IMPLEMENTATION CHECKLIST

### Setup Google Colab
- [ ] Mount Google Drive (jika menggunakan dataset dari Drive)
- [ ] Upload dataset CAD-RADS ke Colab
- [ ] Copy kode `kmeans_implementation.py` ke cell Colab
- [ ] Install required libraries (numpy, pandas, matplotlib, seaborn, scipy)

### Prepare Dataset
- [ ] Load dataset dengan `pd.read_csv()`
- [ ] Cek shape dan column names
- [ ] Cek missing values
- [ ] Tentukan fitur mana yang digunakan (JANGAN gunakan 'Cath')
- [ ] Dokumentasikan alasan pemilihan fitur di laporan

---

## 🚀 IMPLEMENTATION WORKFLOW

### Phase 1: Data Preparation (Week 1)
```python
# ✓ STEP 1: Load & Explore Data
df = pd.read_csv('data_cad_rads.csv')
print(df.shape)
print(df.isnull().sum())
print(df.describe())

# ✓ STEP 2: Clean Data
df_clean = df.dropna()  # atau fillna

# ✓ STEP 3: Select Features
selected_features = [col for col in df.columns if col != 'Cath']
X = df_clean[selected_features].values

# ✓ STEP 4: Normalize Data (PENTING!)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✓ Dokumentasikan: Berapa data points? Berapa features? Min/max values?
```

**Deliverable Week 1:**
- [ ] Dataset sudah clean
- [ ] Jumlah samples dan features terdokumentasi
- [ ] Ada visualisasi data distribution (histogram, scatter)

---

### Phase 2: Determine Optimal K (Week 1-2)
```python
# ✓ STEP 1: Run Elbow Method
k_range = range(2, 11)
sse_values, silhouette_scores, db_indices = elbow_method(X_scaled, k_range)

# ✓ STEP 2: Plot Results
plot_elbow_curve(k_range, sse_values, silhouette_scores, db_indices)

# ✓ STEP 3: Analyze and Choose K
print("K Analysis:")
for i, k in enumerate(k_range):
    print(f"k={k}: SSE={sse_values[i]:.4f}, Silhouette={silhouette_scores[i]:.4f}, DB={db_indices[i]:.4f}")

# Cari "elbow" di SSE plot
# Pilih k dengan Silhouette score paling tinggi
# Pilih k dengan DB Index paling rendah
# DECISION: optimal_k = ???
```

**Decision Making Guide:**
| Metric | Guidance |
|--------|----------|
| **SSE** | Cari titik di mana SSE mulai stabil (elbow) |
| **Silhouette** | Cari nilai tertinggi (ideally > 0.5) |
| **DB Index** | Cari nilai terendah (ideally < 1.0) |

**Deliverable Week 2:**
- [ ] Elbow method plot yang jelas
- [ ] Tabel perbandingan semua metrik untuk k=2 sampai k=10
- [ ] Analisis tertulis mengapa memilih k tersebut
- [ ] Silhouette plot untuk k optimal

---

### Phase 3: Training & Evaluation (Week 2-3)
```python
# ✓ STEP 1: Train Model
optimal_k = 3  # SESUAIKAN BERDASARKAN ANALISIS ANDA
kmeans = KMeansCustom(k=optimal_k, max_iter=100, random_state=42, verbose=True)
labels = kmeans.fit_predict(X_scaled)

# ✓ STEP 2: Evaluate Quality
silhouette_avg, silhouette_values = calculate_silhouette_score(X_scaled, labels)
db_score = davies_bouldin_index(X_scaled, labels, kmeans.centroids)

print(f"Silhouette Score: {silhouette_avg:.4f} ← Interpretasi: ???")
print(f"Davies-Bouldin Index: {db_score:.4f} ← Interpretasi: ???")

# ✓ STEP 3: Analyze Cluster Distribution
unique, counts = np.unique(labels, return_counts=True)
for cluster_id, count in zip(unique, counts):
    percentage = (count / len(labels)) * 100
    print(f"Cluster {cluster_id}: {count} points ({percentage:.1f}%)")

# Apakah distribusi balanced? Ada cluster yang terlalu kecil/besar?

# ✓ STEP 4: Check for Empty Clusters
if len(np.unique(labels)) < optimal_k:
    print("⚠️  Ada cluster yang kosong! Pertimbangkan ganti k atau seed")
```

**Deliverable Week 3:**
- [ ] Silhouette score dan interpretasinya
- [ ] Davies-Bouldin Index dan interpretasinya
- [ ] Distribusi cluster (berapa persen di setiap cluster)
- [ ] Silhouette plot untuk k optimal

---

### Phase 4: Cluster Analysis & Visualization (Week 3)
```python
# ✓ STEP 1: Denormalize Centroids untuk Interpretasi
centroids_original = scaler.inverse_transform(kmeans.centroids)

# ✓ STEP 2: Analyze Each Cluster
print("="*70)
print("KARAKTERISTIK SETIAP CLUSTER")
print("="*70)

for cluster_id in range(optimal_k):
    print(f"\n{'='*70}")
    print(f"CLUSTER {cluster_id}")
    print('='*70)
    
    cluster_mask = labels == cluster_id
    cluster_size = np.sum(cluster_mask)
    cluster_data = X[cluster_mask]
    
    print(f"\nUkuran: {cluster_size} points ({cluster_size/len(labels)*100:.1f}%)")
    
    print(f"\nCentroid (original values):")
    for i, feature in enumerate(selected_features):
        print(f"  {feature}: {centroids_original[cluster_id, i]:.4f}")
    
    print(f"\nCharacteristics (mean ± std):")
    for i, feature in enumerate(selected_features):
        mean = cluster_data[:, i].mean()
        std = cluster_data[:, i].std()
        overall_mean = X[:, i].mean()
        diff = mean - overall_mean
        sign = "↑" if diff > 0 else "↓"
        print(f"  {feature}: {mean:.4f} ± {std:.4f} (overall: {overall_mean:.4f}) {sign}")
    
    # Interpretasi: Apa karakteristik unik cluster ini dibanding cluster lain?
```

**Analysis Questions:**
- Cluster mana yang paling besar? Terkecil?
- Apakah ada fitur yang sangat berbeda antar cluster?
- Bagaimana interpretasi cluster ini dalam domain CAD-RADS?
- Apakah hasil clustering masuk akal secara medis?

**Deliverable Week 3:**
- [ ] Detailed karakterisasi setiap cluster
- [ ] Tabel perbandingan mean values antar cluster
- [ ] Identifikasi fitur pembeda utama antar cluster

---

### Phase 5: Visualization (Week 3-4)
```python
# ✓ STEP 1: 2D Clustering Plot (menggunakan 2 fitur utama)
plot_clusters_2d(X_scaled, labels, kmeans.centroids, 
                title=f"K-Means Clustering (k={optimal_k})")

# ✓ STEP 2: Silhouette Plot
plot_silhouette_analysis(X_scaled, labels, optimal_k)

# ✓ STEP 3: Cluster Size Distribution
fig, ax = plt.subplots(figsize=(10, 6))
unique, counts = np.unique(labels, return_counts=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
ax.bar([f'Cluster {c}' for c in unique], counts, color=colors)
ax.set_ylabel('Number of Points')
ax.set_title(f'Cluster Size Distribution (k={optimal_k})')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ✓ STEP 4: Feature Distribution per Cluster (boxplot)
fig, axes = plt.subplots(1, len(selected_features), figsize=(15, 4))
for idx, feature in enumerate(selected_features):
    data_by_cluster = [X[labels == i, idx] for i in range(optimal_k)]
    axes[idx].boxplot(data_by_cluster, labels=[f'C{i}' for i in range(optimal_k)])
    axes[idx].set_title(f'{feature}')
    axes[idx].set_ylabel('Value')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Deliverable Week 4:**
- [ ] 2D clustering plot yang jelas
- [ ] Silhouette plot
- [ ] Cluster size bar chart
- [ ] Boxplot feature distribution per cluster (minimal 3 plot)

---

### Phase 6: Save Results & Documentation (Week 4)
```python
# ✓ STEP 1: Save Clustering Results
df_clean['Cluster'] = labels
df_clean.to_csv('hasil_clustering.csv', index=False)
print("✓ Hasil clustering disimpan ke 'hasil_clustering.csv'")

# ✓ STEP 2: Save Summary
summary = {
    'Optimal K': optimal_k,
    'Total Samples': len(X_scaled),
    'Total Features': X_scaled.shape[1],
    'Silhouette Score': silhouette_avg,
    'Davies-Bouldin Index': db_score,
    'Iterations to Converge': len(kmeans.history),
    'Features Used': ', '.join(selected_features)
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('clustering_summary.csv', index=False)
print("✓ Summary disimpan ke 'clustering_summary.csv'")

# ✓ STEP 3: Save Code
# Copy seluruh code ke file .py untuk backup

# ✓ STEP 4: Document Everything
# - Catat semua keputusan (k choice, features selection)
# - Catat semua hasil (metrik, interpretasi)
# - Catat semua insights dan temuan
```

**Deliverable Week 4:**
- [ ] File 'hasil_clustering.csv' dengan label cluster
- [ ] File 'clustering_summary.csv' dengan metrics
- [ ] Backup code di folder yang aman
- [ ] Dokumentasi lengkap semua keputusan

---

## 📝 LAPORAN REQUIREMENTS

### Struktur Laporan Minimum

```
LAPORAN TUGAS BESAR K-MEANS CLUSTERING

I. PENDAHULUAN
   A. Background Masalah
   B. Deskripsi Dataset
   C. Tujuan Penelitian
   
II. TINJAUAN PUSTAKA
   A. Definisi K-Means Clustering
   B. Algoritma K-Means (step-by-step)
   C. Kelebihan dan Kekurangan K-Means
   D. Metrik Evaluasi (SSE, Silhouette, DB Index)
   E. Teknik Menentukan K Optimal
   
III. METODOLOGI
   A. Data Preparation
      1. Dataset Description
      2. Feature Selection & Justification
      3. Normalisasi
   B. Elbow Method for K Selection
   C. K-Means Training
   D. Evaluation Metrics
   
IV. HASIL DAN ANALISIS
   A. Elbow Method Results
      - Tabel & plot SSE, Silhouette, DB Index
      - Alasan memilih K
   B. Clustering Results
      - Metrics (Silhouette, DB Index)
      - Cluster Distribution
      - Silhouette Plot
   C. Cluster Characterization
      - Karakterisasi setiap cluster
      - Tabel perbandingan mean values
      - Interpretasi domain
   D. Visualisasi
      - 2D plot clustering
      - Boxplot feature distribution
      - Cluster size distribution
   
V. DISKUSI & ANALISIS LANJUTAN
   A. Apakah hasil clustering sesuai ekspektasi?
   B. Apakah ada cluster yang kurang sesuai?
   C. Bagaimana hasil ini bisa digunakan?
   D. Keterbatasan penelitian
   E. Saran untuk perbaikan
   
VI. KESIMPULAN
   A. Ringkasan temuan utama
   B. Answer to research questions
   C. Implikasi hasil
   
VII. REFERENSI
   - Minimal 5 referensi (jurnal, textbook, paper)
   
VIII. APPENDIX (Opsional)
   - Code snippets penting
   - Additional plots
   - Raw data samples
```

**Estimasi Halaman:** 15-25 halaman

---

## 🎯 GRADING RUBRIC (Self-Check)

### Code Quality (35%)
- [ ] K-Means diimplementasi dari awal (tidak gunakan sklearn.KMeans)
- [ ] Code terstruktur baik dengan comments
- [ ] Ada error handling
- [ ] Hasil reproducible dengan random_state
- [ ] Semua fungsi helper sudah ada (elbow, silhouette, db index, dll)

**Target:** 32-35/35

### Report Quality (35%)
- [ ] Laporan rapi dan terstruktur
- [ ] Penjelasan K-Means jelas dan akurat
- [ ] Data preprocessing terdokumentasi baik
- [ ] Analisis mendalam (bukan hanya narasi hasil)
- [ ] Ada interpretasi domain knowledge
- [ ] Visualisasi berkualitas dan informatif
- [ ] Referensi yang tepat

**Target:** 30-35/35

### Presentation (30%)
- [ ] Slide menarik dan informatif
- [ ] Presentasi lancar dan jelas
- [ ] Menjawab pertanyaan reviewer dengan baik
- [ ] Demo code berjalan lancar
- [ ] Durasi presentasi tepat waktu

**Target:** 27-30/30

**TOTAL MINIMUM:** 89-100 (A)

---

## 🚨 COMMON MISTAKES TO AVOID

### ❌ MISTAKE 1: Menggunakan sklearn KMeans
```python
# SALAH:
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)  # ← TIDAK BOLEH!

# BENAR:
kmeans = KMeansCustom(k=3)  # ← Gunakan implementasi sendiri
```

### ❌ MISTAKE 2: Tidak Normalisasi Data
```python
# SALAH:
kmeans.fit(X)  # ← Data tidak di-scale

# BENAR:
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)  # ← Data sudah di-normalize
```

### ❌ MISTAKE 3: Lupa Gunakan Random State
```python
# SALAH:
kmeans = KMeansCustom(k=3)  # ← Hasil berbeda setiap run

# BENAR:
kmeans = KMeansCustom(k=3, random_state=42)  # ← Reproducible
```

### ❌ MISTAKE 4: Tidak Analyze Hasil
```python
# SALAH: Hanya print labels
print(labels)

# BENAR: Analyze karakteristik cluster
for cluster_id in range(optimal_k):
    cluster_data = X[labels == cluster_id]
    print(f"Cluster {cluster_id} mean: {cluster_data.mean(axis=0)}")
```

### ❌ MISTAKE 5: Menggunakan Cath Feature
```python
# SALAH:
selected_features = df.columns  # ← Termasuk Cath!

# BENAR:
selected_features = [col for col in df.columns if col != 'Cath']
```

### ❌ MISTAKE 6: Tidak Dokumentasi Keputusan
```python
# SALAH: Tidak ada penjelasan
optimal_k = 3

# BENAR: Ada penjelasan di laporan
# "Kami memilih k=3 karena:
#  1. Elbow method menunjukkan elbow point di k=3
#  2. Silhouette score tertinggi di k=3 (0.68)
#  3. DB Index terendah di k=3 (0.45)"
```

---

## 📊 TEMPLATE SUMMARY TABLE

Gunakan tabel ini di laporan:

```
Tabel 1: Metrics untuk berbagai K
┌────────┬──────────┬──────────────┬──────────┐
│   K    │   SSE    │  Silhouette  │ DB Index │
├────────┼──────────┼──────────────┼──────────┤
│   2    │ 1234.56  │    0.45      │  0.67    │
│   3    │  987.65  │    0.68 ★    │  0.45 ★  │
│   4    │  876.54  │    0.62      │  0.52    │
│   5    │  765.43  │    0.55      │  0.61    │
└────────┴──────────┴──────────────┴──────────┘
★ = Selected K
```

```
Tabel 2: Cluster Characteristics
┌─────────────┬──────────┬──────────┬──────────┐
│   Feature   │ Cluster0 │ Cluster1 │ Cluster2 │
├─────────────┼──────────┼──────────┼──────────┤
│    Age      │  45.2    │  52.3    │  38.1    │
│    BMI      │  28.5    │  31.2    │  25.3    │
│ Diabetes    │   0.35   │   0.52   │   0.15   │
└─────────────┴──────────┴──────────┴──────────┘
```

---

## ✅ FINAL SUBMISSION CHECKLIST

### Before Submission
- [ ] Code sudah ditest dan berjalan
- [ ] Dataset sudah di-preprocess
- [ ] K optimal sudah ditentukan dengan analisis mendalam
- [ ] Model sudah di-train
- [ ] Semua metrics sudah dihitung
- [ ] Visualisasi sudah dibuat
- [ ] Hasil sudah dianalisis

### Colab Link
- [ ] Colab shared (public atau allow access)
- [ ] Code lengkap dan rapi
- [ ] Output jelas dan terlihat
- [ ] Dokumentasi inline di code

### Laporan
- [ ] Laporan selesai dan proofread
- [ ] Semua gambar sudah embed (bukan link)
- [ ] Referensi sudah di-format dengan benar
- [ ] Page number & table of contents
- [ ] Ukuran file ≤ 10MB

### Presentasi PPT
- [ ] Slide design profesional dan konsisten
- [ ] Content jelas dan ringkas
- [ ] Visual (chart, plot) berkualitas
- [ ] Durasi 10-15 menit
- [ ] Practice presentasi minimal 1x

### Format Nama File
```
[NOMOR_KELOMPOK]_[KELAS]_KMeans.pptx
[NOMOR_KELOMPOK]_[KELAS]_KMeans.pdf  (laporan)
```

### Link Submission
- Colab link: [paste link here]
- Laporan: [paste link/path here]
- PPT: [paste link/path here]

**DEADLINE:** 6 Juni 2026 pukul 22.00 WIB

---

## 🎓 LEARNING OUTCOMES

Setelah menyelesaikan tugas ini, Anda akan bisa:

1. ✓ Memahami konsep dan algoritma K-Means dengan mendalam
2. ✓ Mengimplementasikan K-Means dari awal tanpa library
3. ✓ Melakukan preprocessing dan normalisasi data
4. ✓ Menentukan jumlah cluster optimal menggunakan multiple methods
5. ✓ Mengevaluasi kualitas clustering dengan berbagai metrics
6. ✓ Menganalisis dan menginterpretasikan hasil clustering
7. ✓ Melakukan visualisasi data dan hasil clustering
8. ✓ Menulis laporan ilmiah yang terstruktur
9. ✓ Mempresentasikan hasil penelitian dengan baik

---

## 💡 BONUS: CHALLENGE IDEAS

Untuk nilai lebih, pertimbangkan:

1. **K-Means++** - Inisialisasi centroid yang lebih smart
2. **Elbow Method Otomatis** - Algorithm untuk detect elbow point
3. **Feature Importance** - Analisis fitur mana yang paling penting
4. **Multiple Runs** - Jalankan berkali-kali, bandingkan hasil
5. **Outlier Detection** - Identifikasi dan handle outliers
6. **Comparison** - Bandingkan dengan Hierarchical Clustering
7. **Domain Analysis** - Interpretasi cluster dalam konteks medis CAD-RADS

---

**GOOD LUCK! 🚀**

Ingat: **Understand first, code second, analyze third!**
