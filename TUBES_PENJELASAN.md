# Penjelasan Lengkap Notebook `TUBES.ipynb`



## 1. Ringkasan Alur Notebook

Urutan kerja notebook adalah sebagai berikut:

1. Import library.
2. Membaca dataset.
3. Exploratory Data Analysis (EDA).
4. Preprocessing dan supervised learning.
5. Evaluasi model supervised.
6. Unsupervised learning dengan K-Means dari awal.
7. Elbow method dan evaluasi kualitas cluster.
8. Visualisasi cluster dan silhouette analysis.
9. Analisis karakteristik cluster.

Urutan ini penting karena bagian clustering bergantung pada hasil preprocessing dari bagian supervised.

## 2. Penjelasan Setiap Bagian

### 2.1 Import Library

Bagian awal notebook memuat library yang dibutuhkan:

- `pandas` dan `numpy` untuk pengolahan data.
- `matplotlib.pyplot` dan `seaborn` untuk visualisasi.
- `warnings` untuk menonaktifkan warning yang tidak penting.
- `sklearn` untuk preprocessing, model klasifikasi, feature selection, dan evaluasi.

Fungsi utama library yang digunakan:

- `train_test_split` untuk membagi data latih dan data uji.
- `RandomForestClassifier`, `MLPClassifier`, dan `GaussianNB` untuk supervised learning.
- `StandardScaler` untuk standarisasi fitur.
- `LabelEncoder` untuk mengubah label kategorikal menjadi numerik.
- `SelectKBest` dan `mutual_info_classif` untuk memilih fitur terbaik.
- `silhouette_score` untuk menilai kualitas cluster.

### 2.2 Membaca Dataset

Dataset dibaca dari file `CADalizadeh.xls` dan disimpan ke variabel `df`.

Tujuan langkah ini adalah memuat data mentah sebelum dianalisis.

### 2.3 Exploratory Data Analysis (EDA)

EDA digunakan untuk memahami data sebelum membuat model.

Informasi yang ditampilkan pada bagian ini meliputi:

- ukuran dataset: jumlah baris dan kolom,
- 5 baris pertama data,
- info tipe data setiap kolom,
- statistik deskriptif,
- jumlah missing value per kolom,
- jumlah fitur numerik dan kategorikal,
- distribusi target `Cath`,
- korelasi antar fitur numerik,
- histogram fitur-fitur penting,
- boxplot fitur terhadap `Cath`.

Fungsi EDA:

- memberi gambaran umum isi dataset,
- membantu melihat data hilang,
- membantu melihat sebaran data,
- membantu memahami hubungan awal antar fitur,
- membantu memilih fitur yang relevan.

### 2.4 Preprocessing untuk Supervised Learning

Sebelum model dilatih, data dibersihkan terlebih dahulu:

- `drop_duplicates()` untuk menghapus baris duplikat.
- `dropna()` untuk menghapus baris yang memiliki nilai kosong.

Setelah itu dilakukan:

- encoding target `Cath` dengan `LabelEncoder`,
- pembentukan `X_full` dari semua kolom numerik kecuali `Cath`,
- penghapusan kolom `No` jika ada, karena hanya berfungsi sebagai identitas.

`X_full` penting karena:

- dipakai sebagai dasar fitur supervised,
- dipakai juga sebagai sumber data untuk clustering.

### 2.5 Seleksi Fitur untuk Supervised Learning

Notebook memakai `SelectKBest(mutual_info_classif)` untuk memilih fitur terbaik dalam memprediksi `Cath`.

Alasan metode ini dipakai:

- `mutual_info_classif` bisa menangkap hubungan non-linear antara fitur dan target,
- fitur yang dipilih cenderung lebih relevan,
- model menjadi lebih sederhana dan lebih fokus pada informasi penting.

Nilai `k_sel` dibuat sebagai:

- `k_sel = min(8, jumlah_fitur)`

Artinya notebook memilih maksimal 8 fitur terbaik, atau lebih sedikit jika fitur numeriknya kurang dari 8.

Hasilnya:

- `X` = fitur terpilih untuk supervised learning,
- `y` = target `Cath`.

### 2.6 Pembagian Data dan Scaling untuk Supervised Learning

Data dibagi menjadi data latih dan data uji menggunakan `train_test_split`.

Parameter `stratify=y` dipakai agar proporsi kelas pada data latih dan uji tetap seimbang.

Setelah itu fitur distandarisasi dengan `StandardScaler`.

Mengapa scaling diperlukan:

- model seperti MLP sensitif terhadap skala fitur,
- perbandingan antar fitur menjadi lebih adil,
- proses pelatihan model menjadi lebih stabil.

### 2.7 Model Supervised yang Dipakai

Notebook melatih tiga model klasifikasi:

1. **RandomForestClassifier**
   - kuat untuk data tabular,
   - bagus untuk pola non-linear,
   - relatif tahan terhadap noise.

2. **MLPClassifier**
   - model jaringan saraf sederhana,
   - fleksibel untuk pola non-linear,
   - membutuhkan scaling dan biasanya butuh tuning lebih lanjut.

3. **GaussianNB**
   - model baseline yang sederhana dan cepat,
   - cocok sebagai pembanding awal,
   - mengasumsikan fitur saling independen.

### 2.8 Evaluasi Supervised Learning

Setiap model dievaluasi menggunakan:

- **Accuracy**: proporsi prediksi yang benar.
- **Precision**: ketepatan prediksi pada kelas tertentu.
- **Recall**: kemampuan model menemukan data aktual.
- **F1-score**: keseimbangan antara precision dan recall.

Notebook juga menampilkan:

- classification report,
- confusion matrix.

Interpretasi umum:

- accuracy saja tidak cukup jika data tidak seimbang,
- confusion matrix membantu melihat jenis kesalahan prediksi,
- F1-score lebih informatif untuk kasus klasifikasi medis atau data dengan distribusi kelas yang tidak merata.

## 3. Penjelasan Unsupervised Learning

### 3.1 Data untuk Clustering

Pada versi notebook saat ini, bagian clustering **dipaksa menggunakan satu atribut saja**, yaitu `Age`.

Artinya:

- tidak ada pilihan memakai semua fitur numerik,
- notebook mengambil kolom `Age` dari `X_full`,
- jika `X_full` belum tersedia, notebook akan menampilkan `NameError`,
- jika kolom `Age` tidak ada di `X_full`, notebook akan menampilkan `KeyError`.

Tujuan pembatasan ini:

- membuat clustering lebih sederhana,
- memudahkan interpretasi hasil,
- cocok untuk presentasi dan laporan dasar.

### 3.2 Scaling untuk Clustering

Sebelum clustering, fitur `Age` distandarisasi dengan `StandardScaler`.

Tujuannya:

- jarak Euclidean menjadi lebih bermakna,
- centroid tidak dipengaruhi skala besar-kecil data,
- hasil clustering lebih stabil.

### 3.3 K-Means dari Awal

Notebook mengimplementasikan K-Means secara manual melalui fungsi `kmeans_simple`.

Langkah algoritmanya:

1. Pilih centroid awal secara acak.
2. Hitung jarak setiap titik ke setiap centroid.
3. Tentukan cluster berdasarkan centroid terdekat.
4. Hitung centroid baru sebagai rata-rata anggota cluster.
5. Ulangi sampai centroid tidak berubah signifikan atau iterasi maksimum tercapai.

Mengapa ini penting:

- memenuhi syarat tugas bahwa K-Means harus dibuat dari awal,
- membantu memahami cara kerja algoritma,
- memudahkan penjelasan saat presentasi.

### 3.4 Elbow Method dan Pemilihan k

Notebook mencoba beberapa nilai `k` dalam rentang:

- dari 2 sampai `min(10, n-1)`.

Untuk setiap nilai `k`, dihitung:

- **SSE** (Sum of Squared Errors),
- **Silhouette Score**,
- **Davies-Bouldin Index**.

Fungsi setiap metrik:

- **SSE**: semakin kecil semakin baik, tetapi biasanya turun saat `k` diperbesar.
- **Silhouette Score**: semakin besar semakin baik.
- **Davies-Bouldin Index**: semakin kecil semakin baik.

Strategi pemilihan `optimal_k` di notebook:

1. pilih `k` dengan Silhouette tertinggi jika ada,
2. jika Silhouette tidak valid, pilih `k` dengan Davies-Bouldin terkecil.

### 3.5 Visualisasi Elbow

Grafik SSE dipakai untuk melihat titik siku atau elbow.

Interpretasinya:

- jika penurunan SSE mulai melambat, maka nilai `k` setelah titik tersebut biasanya kurang efisien,
- titik elbow membantu memilih jumlah cluster yang seimbang antara sederhana dan akurat.

### 3.6 Training K-Means dengan k Optimal

Setelah `optimal_k` dipilih, notebook menjalankan K-Means lagi untuk mendapatkan:

- label cluster akhir,
- centroid akhir.

Hasil ini digunakan untuk analisis dan visualisasi berikutnya.

### 3.7 Distribusi Cluster

Notebook menampilkan jumlah data pada setiap cluster.

Tujuan analisis ini:

- melihat apakah cluster seimbang atau tidak,
- mengetahui apakah ada cluster yang terlalu dominan,
- membantu menilai kewajaran hasil clustering.

### 3.8 Visualisasi Hasil Clustering

Karena clustering hanya memakai satu fitur (`Age`), visualisasi dibuat dalam bentuk 1D.

Cara visualisasi:

- titik data digambar pada sumbu-x sesuai nilai `Age` yang sudah distandarisasi,
- sumbu-y diberi jitter kecil agar titik tidak saling menumpuk,
- centroid digambar sebagai marker `X` pada y = 0.

Tujuan jitter:

- membuat titik mudah terlihat,
- tidak mengubah makna data,
- hanya membantu visualisasi.

### 3.9 Silhouette Analysis

Notebook menghitung `silhouette_samples` dan `silhouette_score` untuk cluster akhir.

Makna silhouette:

- nilai mendekati 1: cluster sangat baik,
- nilai sekitar 0: titik berada di perbatasan cluster,
- nilai negatif: titik mungkin lebih cocok ke cluster lain.

Notebook juga menampilkan silhouette plot agar kualitas tiap cluster terlihat lebih jelas.

### 3.10 Analisis Karakteristik Cluster

Pada akhir bagian clustering, notebook menampilkan ringkasan statistik tiap cluster.

Untuk setiap cluster, ditampilkan:

- jumlah anggota cluster,
- persentase anggota cluster,
- mean dan standar deviasi fitur.

Karena fitur clustering hanya `Age`, interpretasi cluster menjadi lebih sederhana:

- cluster merepresentasikan kelompok usia yang berbeda,
- centroid menunjukkan nilai rata-rata usia pada cluster tersebut.

## 4. Penjelasan Konsep Penting untuk PPT

Bagian ini bisa langsung dipakai sebagai isi slide atau narasi presentasi.

### Slide 1: Tujuan Penelitian

Notebook ini bertujuan memprediksi `Cath` dengan supervised learning dan menemukan pola kelompok data dengan K-Means.

### Slide 2: Sumber Data

Dataset dibaca dari file Excel `CADalizadeh.xls` dan dianalisis menggunakan Python.

### Slide 3: EDA

EDA dilakukan untuk memahami struktur data, melihat missing value, dan mengamati distribusi fitur serta target.

### Slide 4: Supervised Learning

Target `Cath` diprediksi menggunakan tiga model klasifikasi: Random Forest, MLP, dan Naive Bayes.

### Slide 5: Seleksi Fitur

Fitur terbaik dipilih memakai `SelectKBest(mutual_info_classif)` agar model fokus pada fitur yang paling relevan terhadap target.

### Slide 6: Evaluasi Model

Model dibandingkan dengan accuracy, precision, recall, F1-score, confusion matrix, dan classification report.

### Slide 7: Unsupervised Learning

Clustering dilakukan dengan K-Means yang dibuat dari awal, tanpa library K-Means siap pakai.

### Slide 8: Data Clustering

Notebook saat ini memakai satu fitur saja, yaitu `Age`, agar hasil clustering mudah dijelaskan.

### Slide 9: Menentukan k

Nilai `k` dipilih menggunakan elbow method, silhouette score, dan Davies-Bouldin index.

### Slide 10: Hasil Clustering

Hasil akhir divisualisasikan dalam bentuk sebaran cluster, jumlah anggota cluster, silhouette plot, dan ringkasan karakteristik cluster.

## 5. Alasan Setiap Langkah Dipakai

### Mengapa data dibersihkan?

Karena data duplikat dan data kosong dapat mengganggu proses training dan evaluasi.

### Mengapa fitur dipilih?

Karena tidak semua fitur sama pentingnya untuk memprediksi `Cath`.

### Mengapa scaling diperlukan?

Karena model berbasis jarak dan model neural network sangat dipengaruhi oleh skala data.

### Mengapa K-Means ditulis manual?

Karena tugas meminta implementasi dari awal, sehingga algoritma tidak boleh langsung memakai library K-Means.

### Mengapa clustering hanya memakai `Age`?

Karena notebook diarahkan untuk analisis yang sederhana, mudah dijelaskan, dan cocok untuk kebutuhan presentasi.

## 6. Interpretasi Hasil yang Mungkin Muncul

### Jika Random Forest paling baik

Ini berarti hubungan antara fitur dan `Cath` cukup kuat dan model ensemble mampu menangkap pola tersebut dengan baik.

### Jika MLP kurang stabil

Biasanya karena model neural network membutuhkan tuning tambahan seperti jumlah neuron, learning rate, atau iterasi.

### Jika Naive Bayes lebih rendah

Hal ini wajar karena model ini memiliki asumsi independensi fitur yang sering tidak terpenuhi pada data nyata.

### Jika silhouette score rendah

Ini menunjukkan cluster tidak terlalu terpisah dengan baik, atau data memang tidak memiliki struktur cluster yang kuat pada atribut yang dipilih.

### Jika cluster tidak seimbang

Artinya satu cluster bisa terlalu dominan, sehingga nilai `k` atau atribut yang dipakai perlu ditinjau ulang.

## 7. Kelebihan dan Keterbatasan Notebook

### Kelebihan

- Alur notebook runtut dan mudah diikuti.
- Supervised learning memakai seleksi fitur untuk mengurangi fitur yang tidak penting.
- K-Means ditulis dari awal sesuai syarat tugas.
- Evaluasi model cukup lengkap.
- Clustering 1D dengan `Age` mudah dijelaskan untuk PPT.

### Keterbatasan

- `dropna()` dapat membuang banyak data jika missing value cukup besar.
- Clustering hanya memakai satu atribut, sehingga pola multivariat tidak terlihat.
- K-Means sensitif terhadap inisialisasi centroid awal.
- Model supervised belum dioptimasi dengan tuning parameter.

## 8. Rangkuman Singkat untuk Laporan

Notebook ini membandingkan pendekatan supervised dan unsupervised pada dataset CAD-RADS. Pada bagian supervised, target `Cath` diprediksi menggunakan Random Forest, MLP, dan Naive Bayes setelah preprocessing, encoding target, standarisasi, dan seleksi fitur dengan mutual information. Pada bagian unsupervised, notebook menerapkan K-Means dari awal dan saat ini memaksa clustering menggunakan satu atribut yaitu `Age`. Kualitas cluster dievaluasi menggunakan SSE, Silhouette Score, dan Davies-Bouldin Index, lalu divisualisasikan agar mudah dipahami.

## 9. Kalimat Siap Pakai untuk Presentasi

Berikut contoh narasi singkat yang bisa langsung dibacakan saat PPT:

"Pada penelitian ini, dataset CAD-RADS dianalisis menggunakan dua pendekatan. Pertama, supervised learning digunakan untuk memprediksi label `Cath` dengan beberapa model klasifikasi. Kedua, unsupervised learning digunakan untuk menemukan pola kelompok data menggunakan K-Means yang dibuat dari awal. Untuk memudahkan interpretasi, clustering pada versi notebook ini difokuskan hanya pada atribut `Age`. Hasil clustering dievaluasi menggunakan SSE, Silhouette Score, dan Davies-Bouldin Index agar jumlah cluster yang dipilih lebih objektif."

## 10. Kesimpulan

Notebook ini sudah berisi alur lengkap dari pembacaan data, EDA, preprocessing, supervised learning, evaluasi model, hingga clustering K-Means dari awal. Versi terakhir notebook menekankan penggunaan `Age` saja pada clustering sehingga hasilnya lebih sederhana, mudah dianalisis, dan lebih mudah dijelaskan dalam presentasi maupun laporan.
