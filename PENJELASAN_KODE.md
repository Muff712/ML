# Penjelasan Kode: Supervised & Unsupervised pada Dataset CAD-RADS

Dokumen ini menjelaskan kode yang ada di repository Anda (khususnya `Kode_Lengkap_Supervised_Unsupervised.py` dan `TUBES.ipynb`) secara ringkas, jelas, dan mudah dipahami. Tujuan: menjelaskan langkah-langkah yang dilakukan, alasan pemilihan teknik, dan mengapa hasil yang diperoleh bisa seperti itu.

**Catatan**: gunakan link berkas di workspace untuk melihat kode: [Kode_Lengkap_Supervised_Unsupervised.py](Kode_Lengkap_Supervised_Unsupervised.py) dan [TUBES.ipynb](TUBES.ipynb).

---

## 1. Gambaran Umum
- Supervised: klasifikasi target `Cath` (label biner/kategori). Pipeline: pembersihan data -> encoding target -> seleksi fitur -> split -> scaling -> melatih beberapa model -> evaluasi metrik.
- Unsupervised: segmentasi/pengelompokan seluruh data (kecuali kolom `Cath`) menggunakan K-Means (implementasi sederhana) dan evaluasi kualitas clustering (SSE, Silhouette, Davies-Bouldin).

## 2. Persiapan Data dan Preprocessing
1. Menghapus duplikat (`drop_duplicates()`) dan baris yang memiliki nilai hilang (`dropna()`).
   - Kenapa: mempermudah pipeline dan menghindari kegagalan fungsi yang tidak menangani NaN.
   - Catatan: menghapus baris NaN sederhana tapi dapat mengurangi ukuran dataset — jika banyak NaN, sebaiknya imputasi.

2. Target encoding: `LabelEncoder()` pada `Cath`.
   - Kenapa: model sklearn mengharapkan label numerik.

3. Memisahkan fitur numerik penuh `X_full` (untuk unsupervised) dan melakukan seleksi fitur untuk supervised.

## 3. Seleksi Fitur untuk Supervised
- Metode: `SelectKBest(mutual_info_classif, k=k_sel)` di mana `k_sel = min(8, n_features)`.
- Mutual information (MI): mengukur ketergantungan (bisa non-linear) antara setiap fitur dan target. MI berguna untuk memilih fitur yang paling informatif untuk klasifikasi.
- Mengapa seleksi fitur: mengurangi dimensi, mengurangi kebisingan, mempercepat pelatihan, dan sering meningkatkan performa model sederhana.

Interpretasi: fitur yang dipilih adalah yang menunjukkan hubungan paling kuat dengan `Cath`. Jika hasil klasifikasi buruk, bisa berarti: fitur tidak informatif, data imbalance, atau fitur butuh transformasi.

## 4. Pipeline Supervised
Langkah-langkah:
- Ambil `X` hanya dari fitur terpilih.
- Split data: `train_test_split(..., stratify=y)` (menjaga proporsi kelas).
- Standarisasi: `StandardScaler()` (penting untuk MLP dan Naive Bayes).
- Model yang dilatih: `RandomForestClassifier`, `MLPClassifier`, `GaussianNB`.
- Evaluasi: `accuracy`, `precision`, `recall`, `f1` (averaged weighted).

Mengapa model ini:
- RandomForest: model ensemble non-parametrik yang kuat terhadap fitur yang tidak diskalakan, baik untuk data tabular.
- MLP: jaringan saraf sederhana, butuh scaling dan hyperparameter tuning.
- NaiveBayes: cepat, asumsi independensi fitur (sering dilanggar), jadi performanya bisa lebih rendah.

Mengapa hasil bisa berbeda antara model:
- RandomForest menangkap interaksi fitur dan non-linearitas tanpa banyak tuning.
- MLP memerlukan lebih banyak data, tuning, dan scaling; jika dataset kecil, dapat overfit atau underfit.
- NaiveBayes peka pada distribusi fitur dan korelasi antar fitur.

Interpretasi hasil metrik:
- Accuracy: proporsi prediksi benar.
- Precision/Recall/F1: lebih informatif untuk dataset imbalance.
  - Precision tinggi: prediksi positif lebih akurat.
  - Recall tinggi: model menangkap banyak positif aktual.
  - F1: trade-off precision/recall.

Jika metrik rendah:
- Kemungkinan fitur kurang informatif untuk membedakan kelas.
- Dataset imbalance atau ukuran sampel kecil.
- Model perlu tuning (mis. `n_estimators`, `max_depth` untuk RF; arsitektur/regularisasi untuk MLP).
- Perlu engineering fitur (transformasi, polynomial, interaksi).

## 5. Pipeline Unsupervised (K-Means)
Langkah-langkah utama di notebook:
- Mengambil semua fitur numerik kecuali `Cath` (`cluster_df`).
- Standarisasi dengan `StandardScaler()`.
- Implementasi K-Means sederhana (`kmeans_simple`) yang:
  - Menginisialisasi centroid dengan memilih k sampel acak,
  - Meng-assign label berdasarkan jarak Euclidean,
  - Meng-update centroid sebagai rata-rata cluster,
  - Mengulang sampai konvergen atau mencapai `max_iter`.
- Evaluasi jumlah cluster melalui Elbow method (SSE), dan juga silhouette score / Davies-Bouldin (jika dihitung).

Mengapa memilih semua fitur (bukan fitur terpilih untuk `Cath`):
- Tujuan clustering berbeda: menangkap struktur alami data, bukan memprediksi `Cath`. Penggunaan semua fitur numerik (kecuali target) membantu menemukan grup berdasarkan seluruh profil.

Interprestasi metrik clustering:
- SSE (Sum of Squared Errors): turun ketika k bertambah; Elbow point memberi trade-off antara SSE dan jumlah cluster.
- Silhouette Score: berkisar -1 s.d. +1; lebih tinggi lebih baik (≥0.5 umumnya baik).
- Davies-Bouldin: lebih kecil lebih baik.

Mengapa hasil clustering bisa terlihat seperti itu:
- Jika fitur kurang membedakan, cluster bisa tidak jelas (silhouette rendah).
- Skala fitur: scaling penting — tanpa scaling, fitur berdimensi besar mendominasi.
- Inisialisasi centroid acak: bisa mempengaruhi hasil; coba beberapa inisialisasi (restarts) untuk stabilitas.
- Outlier: dapat memengaruhi centroid dan SSE.

## 6. Contoh Penyebab Hasil Spesifik
- RandomForest performa terbaik: karena mampu menangkap non-linearitas dan interaksi tanpa memerlukan fitur yang sangat relevan satu per satu.
- NaiveBayes performa buruk: asumsi independensi terlanggar, atau fitur tidak berdistribusi normal.
- Silhouette rendah (cluster buruk): fitur tidak memisahkan populasi, atau k terlalu besar/terlalu kecil.

## 7. Cara Menjalankan Kode
Di terminal (virtualenv aktif):

```bash
python Kode_Lengkap_Supervised_Unsupervised.py
```

atau untuk menjalankan notebook (di lingkungan dengan Jupyter):

```bash
jupyter notebook TUBES.ipynb
# lalu jalankan sel-selnya di UI
```

Untuk eksekusi batch (menjalankan seluruh notebook dan menyimpan output):

```bash
python -m nbconvert --to notebook --execute TUBES.ipynb --inplace
```

## 8. Rekomendasi Untuk Meningkatkan Hasil
- Tangani nilai hilang dengan imputasi (mean/median/KNN) bukan `dropna()` bila banyak NaN.
- Lakukan eksplorasi distribusi per fitur, transformasi (log/Box-Cox) bila fitur skewed.
- Tambah feature engineering: kombinasi fitur, rasio, atau domain-specific features.
- Lakukan hyperparameter tuning (GridSearch/RandomizedSearch/CV) untuk `RandomForest` dan `MLP`.
- Untuk clustering: coba `kmeans++` inisialisasi, multi-restarts, atau gunakan algoritma lain (DBSCAN, GaussianMixture).
- Jika dataset besar/dimensi tinggi: pertimbangkan PCA sebelum clustering.

## 9. Ringkasan Singkat
- Supervised pipeline memilih fitur yang paling informatif untuk memprediksi `Cath` menggunakan mutual information, lalu membandingkan tiga model klasik.
- Unsupervised pipeline mengelompokkan pasien berdasarkan seluruh fitur numerik (tanpa target) menggunakan K-Means sederhana dan menilai kualitas clustering.
- Hasil bergantung pada kualitas fitur, ukuran dataset, penanganan NaN, scaling, dan hyperparameter.

---

Jika Anda mau, saya bisa:
- Jalankan notebook dan kirim ringkasan hasil metrik + rekomendasi tindakan selanjutnya; atau
- Tambahkan contoh visual/interaktif pada file markdown; atau
- Pindahkan isi penjelasan ini ke dalam sel markdown di `TUBES.ipynb` tanpa mengubah struktur lainnya.

Pilih opsi selanjutnya atau minta revisi penjelasan bila ada bagian yang ingin diperluas.