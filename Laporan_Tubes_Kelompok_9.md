# LAPORAN TUGAS BESAR MACHINE LEARNING
## KLASIFIKASI PENYAKIT JANTUNG KORONER (CAD) MENGGUNAKAN METODE SUPERVISED & UNSUPERVISED LEARNING PADA DATASET CAD ALIZADEH

---

### **BAB I: PENDAHULUAN**

#### **1.1 Latar Belakang**
Penyakit Jantung Koroner (Coronary Artery Disease/CAD) merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini terhadap penyakit ini sangat penting untuk meningkatkan peluang keberhasilan pengobatan dan menyelamatkan nyawa pasien. Seiring berkembangnya teknologi informasi, teknik data mining dan machine learning dapat dimanfaatkan untuk membantu para praktisi medis dalam melakukan diagnosis penyakit jantung secara cepat dan akurat berdasarkan karakteristik klinis pasien.

Dataset CAD Alizadeh merupakan salah satu dataset populer yang banyak digunakan dalam penelitian klasifikasi penyakit jantung. Dataset ini terdiri dari berbagai atribut klinis seperti informasi demografis, gejala fisik, hasil elektrokardiogram (EKG), hingga tes laboratorium kimia darah. Dengan memanfaatkan algoritma machine learning, pola hubungan tersembunyi antarfirur klinis tersebut dapat diekstraksi untuk memprediksi hasil diagnosis pemeriksaan kateterisasi jantung (target: `Cath`).

Pada penelitian ini, dilakukan dua pendekatan machine learning utama, yaitu:
1. **Supervised Learning (Klasifikasi)**: Menggunakan tiga algoritma pembanding yaitu **Random Forest**, **Multi-Layer Perceptron (MLP)**, dan **Naive Bayes** untuk mengklasifikasikan hasil pemeriksaan kateterisasi jantung pasien menjadi kelas `Cad` (Jantung Koroner) atau `Normal`.
2. **Unsupervised Learning (Clustering)**: Menggunakan algoritma **K-Means** secara manual (dari awal tanpa menggunakan library scikit-learn) untuk mengelompokkan karakteristik pasien berdasarkan atribut **`Age`** (usia) tanpa mempertimbangkan label target kelas `Cath`.

#### **1.2 Rumusan Masalah**
1. Bagaimana menerapkan model klasifikasi supervised menggunakan algoritma Random Forest, MLP, dan Naive Bayes pada dataset CAD Alizadeh?
2. Bagaimana mengimplementasikan algoritma K-Means secara manual untuk pengelompokkan data berdasarkan karakteristik usia?
3. Bagaimana perbandingan performa ketiga model klasifikasi serta kualitas clustering K-Means yang dihasilkan?

#### **1.3 Tujuan Penelitian**
1. Melakukan klasifikasi data CAD Alizadeh dengan menggunakan algoritma Random Forest, MLP, dan Naive Bayes.
2. Mengimplementasikan algoritma K-Means tanpa library siap pakai untuk melakukan segmentasi data pasien.
3. Mengevaluasi secara mendalam kinerja model klasifikasi dan kualitas kluster yang terbentuk.

#### **1.4 Manfaat Penelitian**
Penelitian ini diharapkan dapat memberikan kontribusi akademik dan praktis mengenai penerapan machine learning di bidang kesehatan (*healthcare analytics*), khususnya dalam membandingkan efektivitas model klasifikasi dan memahami segmentasi pasien jantung berdasarkan profil usia secara otomatis.

---

### **BAB II: DASAR TEORI**

#### **2.1 Data Mining**
Data mining adalah proses menemukan pola, asosiasi, perubahan, anomali, dan struktur penting yang tersembunyi dari sekumpulan data berukuran besar. Proses ini menggabungkan metode dari statistik, machine learning, dan manajemen database.

#### **2.2 Machine Learning**
Machine learning merupakan cabang dari kecerdasan buatan (*Artificial Intelligence*) yang fokus pada pengembangan sistem yang mampu belajar secara mandiri dari data historis guna menghasilkan keputusan atau prediksi tanpa perlu diprogram secara eksplisit.

#### **2.3 Model Klasifikasi Supervised**
1. **Random Forest**: Algoritma berbasis *ensemble learning* yang terdiri dari banyak pohon keputusan (*decision tree*). Setiap pohon dilatih menggunakan sampel *bootstrap* dari data latih, dan prediksi akhir ditentukan melalui mekanisme *voting* (klasifikasi) atau rata-rata (regresi). Random Forest sangat tangguh untuk data tabular klinis karena meminimalkan risiko *overfitting*.
2. **Multi-Layer Perceptron (MLP)**: Model Jaringan Saraf Tiruan (*Artificial Neural Network*) yang terdiri dari minimal tiga lapis neuron: *input layer*, *hidden layer*, dan *output layer*. MLP memanfaatkan teknik *backpropagation* untuk melatih bobot antarsaraf berdasarkan fungsi aktivasi non-linear.
3. **Naive Bayes**: Metode klasifikasi statistik berbasis Teorema Bayes yang mengasumsikan independensi (saling lepas) yang kuat di antara setiap atribut prediktor (*naive assumption*).

#### **2.4 Clustering**
Clustering merupakan salah satu teknik dalam *unsupervised learning* yang bertujuan mengelompokkan sekumpulan objek ke dalam beberapa kelompok (cluster) sehingga objek-objek dalam satu kelompok memiliki kemiripan yang tinggi, sedangkan objek antarkelompok memiliki perbedaan yang signifikan.

#### **2.5 K-Means**
K-Means merupakan algoritma clustering non-hierarki yang membagi data menjadi $k$ buah kelompok. Algoritma ini meminimalkan jarak antara titik data dengan centroid (pusat kluster) masing-masing kelompok yang dihitung menggunakan jarak Euclidean. Tahapan K-Means meliputi:
1. Inisialisasi $k$ centroid awal secara acak.
2. Hitung jarak setiap data ke seluruh centroid.
3. Kelompokkan data ke cluster terdekat.
4. Perbarui nilai centroid berdasarkan rata-rata anggota cluster.
5. Ulangi proses hingga centroid konvergen (tidak berubah).

#### **2.6 Evaluasi Klasifikasi**
Kinerja klasifikasi diukur menggunakan beberapa parameter utama:
* **Accuracy**: Persentase prediksi benar terhadap total seluruh data.
* **Precision**: Proporsi prediksi positif yang benar-benar positif.
* **Recall / Sensitivity**: Proporsi data positif aktual yang berhasil diidentifikasi.
* **F1-Score**: Rata-rata harmonik antara precision dan recall yang menunjukkan keseimbangan performa model.
* **Confusion Matrix**: Matriks visualisasi yang menampilkan perbandingan antara label aktual dan label hasil prediksi model.

#### **2.7 Evaluasi Clustering**
Kualitas hasil clustering dievaluasi dengan:
* **Elbow Method (SSE)**: Mengukur jumlah kuadrat jarak antara anggota cluster ke centroid masing-masing. Titik siku (*elbow*) pada kurva SSE terhadap $k$ menunjukkan jumlah cluster optimal.
* **Silhouette Score**: Mengukur seberapa dekat suatu objek dengan objek lain di clusternya dibandingkan dengan kluster terdekat lainnya (berkisar antara -1 hingga 1).
* **Davies-Bouldin Index (DBI)**: Mengukur kemiripan antar-kluster dengan membandingkan jarak dalam kluster dengan jarak antarkluster. Nilai DBI yang semakin kecil mengindikasikan kualitas pembagian kluster yang semakin baik.

---

### **BAB III: METODOLOGI PENELITIAN**

#### **3.1 Dataset**
Dataset yang digunakan adalah **CAD Alizadeh** (`CADalizadeh.xls`) yang terdiri dari **303 baris data** dan **56 kolom atribut** klinis pasien, dengan rincian 35 fitur bertipe numerik, 20 fitur bertipe kategorikal, dan 1 fitur target bernama `Cath` (dengan label kelas `Cad` dan `Normal`).

#### **3.2 Tahapan Penelitian**
Metodologi dalam penelitian ini dirancang secara sistematis melalui alur kerja berikut:
1. **Pengumpulan Data**: Membaca dataset dari berkas Excel.
2. **Exploratory Data Analysis (EDA)**: Menganalisis dimensi data, tipe data, missing value, korelasi, dan visualisasi distribusi.
3. **Pembersihan Data (*Data Cleaning*)**: Menghapus baris duplikat dan baris dengan nilai kosong (*missing values*).
4. **Target Encoding**: Mengubah kelas target `Cath` menjadi nilai biner numerik menggunakan `LabelEncoder` (0 = `Cad`, 1 = `Normal`).
5. **Seleksi Fitur (Supervised)**: Memilih 8 fitur klinis terbaik menggunakan metode `SelectKBest` dengan basis informasi mutual informasi (`mutual_info_classif`).
6. **Pembagian Dataset**: Membagi dataset menjadi 75% data latih dan 25% data uji secara terstratifikasi (*stratified split*).
7. **Normalisasi Data**: Melakukan standardisasi fitur menggunakan `StandardScaler` agar distribusi data memiliki rata-rata 0 dan variansi 1.
8. **Pelatihan Model Supervised**: Melatih model Random Forest, MLP, dan Naive Bayes pada data latih.
9. **Evaluasi Klasifikasi**: Mengevaluasi performa prediksi model menggunakan metrik akurasi, presisi, recall, F1-Score, dan Confusion Matrix pada data uji.
10. **Implementasi K-Means Manual (Unsupervised)**: Memisahkan atribut `Age`, melakukan standardisasi, melatih algoritma K-Means manual, dan mengevaluasi jumlah kluster optimal $k$ menggunakan Elbow Method, Silhouette Score, dan Davies-Bouldin Index.
11. **Analisis Karakteristik Cluster**: Mengidentifikasi rata-rata nilai usia asli pasien pada kluster-kluster yang terbentuk.

#### **3.3 Preprocessing**
* **Drop Duplicates & Dropna**: Memastikan kualitas data optimal tanpa baris ganda dan tanpa data yang hilang.
* **StandardScaler**: Digunakan untuk menormalkan skala data numerik sehingga fitur berdimensi besar tidak mendominasi perhitungan jarak Euclidean dalam K-Means maupun optimasi gradien dalam MLP.

#### **3.4 Arsitektur Sistem**
* **Alur Supervised**:
  ```
  Input Data -> Pembersihan & Encoding -> Seleksi Fitur (SelectKBest) -> Split (75:25) -> Standarisasi -> Fit & Predict (Random Forest / MLP / Naive Bayes) -> Evaluasi
  ```
* **Alur Unsupervised**:
  ```
  Input Data -> Ekstraksi Atribut 'Age' -> Standarisasi -> K-Means Manual -> Evaluasi Kluster (SSE / Silhouette / DBI) -> Visualisasi & Analisis Profil
  ```

#### **3.5 Implementasi Algoritma K-Means Manual**
Sesuai ketentuan, algoritma K-Means dirancang secara manual menggunakan pustaka `NumPy` tanpa melibatkan `sklearn.cluster.KMeans`. Langkah teknisnya adalah:
1. Centroid awal ($k$ buah) dipilih secara acak dari data `Age` yang telah distandarisasi.
2. Jarak Euclidean dihitung sebagai selisih absolut antara tiap titik data dengan setiap centroid:
   $$D = \sqrt{(x - c)^2} = |x - c|$$
3. Setiap titik data dikelompokkan ke dalam kluster dengan centroid terdekat.
4. Centroid baru dihitung dengan mengambil rata-rata aritmatika dari semua titik data yang berada di kluster tersebut.
5. Proses diulangi hingga centroid konvergen (perubahan centroid di bawah toleransi $1 \times 10^{-4}$) atau mencapai iterasi maksimum (100 iterasi).

---

### **BAB IV: HASIL DAN PEMBAHASAN**

#### **4.1 Statistik Deskriptif**
Dataset terdiri atas 303 pasien dengan visualisasi menunjukkan sebaran data yang lengkap tanpa adanya *missing values* (nilai kosong) di seluruh kolom. Usia pasien (`Age`) berkisar antara 30 hingga 86 tahun dengan rata-rata usia sekitar 58.9 tahun dan standar deviasi sebesar 10.39 tahun.

#### **4.2 Preprocessing Data**
* Setelah pembersihan data, tidak ditemukan baris duplikat maupun baris bernilai kosong, sehingga ukuran data tetap 303 sampel.
* Label target `Cath` diubah menggunakan `LabelEncoder` di mana `Cad` dikodekan sebagai **0** dan `Normal` dikodekan sebagai **1**.
* Melalui `SelectKBest(mutual_info_classif, k=8)`, terpilih 8 fitur prediktor terbaik untuk klasifikasi target `Cath`, yaitu: **`Age`**, **`Typical Chest Pain`**, **`Function Class`**, **`St Elevation`**, **`FBS`** (kadar gula darah puasa), **`Lymph`**, **`EF-TTE`**, dan **`Region RWMA`**.

#### **4.3 Pembagian Dataset**
Data dipisah secara terstratifikasi (agar rasio kelas tetap seimbang) dengan pembagian:
* **75% Data Latih (Train)**: 227 sampel
* **25% Data Uji (Test)**: 76 sampel (terdiri atas 54 sampel kelas `Cad` dan 22 sampel kelas `Normal`)

#### **4.4 Implementasi Model Klasifikasi**
Ketiga model klasifikasi dilatih menggunakan data latih yang telah distandarisasi dan diuji pada 76 data uji. Model dijalankan menggunakan parameter dasar bawaan pustaka scikit-learn.

#### **4.5 Evaluasi Model Klasifikasi**
Berdasarkan hasil eksekusi model pada data uji, diperoleh metrik performa sebagai berikut:

| Model Klasifikasi | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | **0.8158 (81.58%)** | 0.8185 (81.85%) | 0.8158 (81.58%) | 0.7965 (79.65%) |
| **MLP (Neural Network)** | **0.7895 (78.95%)** | 0.7793 (77.93%) | 0.7895 (78.95%) | 0.7784 (77.84%) |
| **Naive Bayes (GaussianNB)**| **0.5395 (53.95%)** | 0.7872 (78.72%) | 0.5395 (53.95%) | 0.5368 (53.68%) |

**Analisis Perbandingan Kinerja Ketiga Model**:
1. **Random Forest (Performa Terbaik)**: Algoritma *ensemble* Random Forest menghasilkan akurasi tertinggi sebesar **81.58%**. Model ini sangat unggul dalam memprediksi kelas mayoritas (`0` / Cad) dengan nilai *recall* mencapai **96%** (52 dari 54 data uji berhasil terdeteksi), meskipun sensitivitas untuk kelas minoritas (`1` / Normal) cenderung lebih rendah (**45%**). Random Forest sangat tangguh untuk dataset klinis terstruktur karena struktur pohon keputusannya secara alami dapat memodelkan interaksi non-linear antarafirur tanpa perlu tuning hyperparameter yang intensif.
2. **MLP (Performa Menengah)**: MLP menghasilkan akurasi sebesar **78.95%**. Performanya sedikit berada di bawah Random Forest namun memiliki sensitivitas prediksi kelas minoritas (`1` / Normal) yang lebih baik, yaitu dengan nilai *recall* sebesar **50%**. Sebagai model Jaringan Saraf Tiruan, MLP membutuhkan data latih dalam jumlah besar untuk mengoptimalkan bobot sinapsisnya. Keterbatasan jumlah sampel (hanya 303 baris) menjadi faktor utama MLP belum bisa melampaui Random Forest secara maksimal.
3. **Naive Bayes (Performa Terendah)**: Naive Bayes memiliki performa paling buruk dengan penurunan akurasi drastis hingga **53.95%**. Model ini memprediksi hampir semua data uji sebagai kelas `1` / Normal (menghasilkan *recall* kelas `1` sebesar **95%** tetapi *recall* kelas `0` jatuh ke angka **37%**). Kegagalan model ini disebabkan oleh asumsi independensi fitur (*naive assumption*). Fitur medis dalam kehidupan nyata, seperti usia, kadar gula darah (`FBS`), nyeri dada (`Typical Chest Pain`), dan kinerja jantung (`EF-TTE`), saling terikat secara patofisiologis. Pelanggaran terhadap asumsi independensi ini mendistorsi perhitungan probabilitas posterior Naive Bayes secara signifikan.

#### **4.6 Implementasi K-Means Tanpa Library**
Proses pengelompokkan data dilakukan pada data satu dimensi dengan mengambil atribut **`Age`** dari matriks `X_full`. Data distandarisasi terlebih dahulu, kemudian inisialisasi centroid dan pembaruan kluster dijalankan secara manual. Hasil iterasi menunjukkan algoritma K-Means manual berhasil konvergen pada iterasi ke-6 untuk nilai $k=2$.

#### **4.7 Evaluasi Hasil Clustering**
Pengujian jumlah kluster dilakukan pada rentang $k = 2$ hingga $k = 10$. Ringkasan evaluasi metrik clustering disajikan pada tabel di bawah ini:

| Jumlah Cluster ($k$) | Sum of Squared Error (SSE) | Silhouette Score | Davies-Bouldin Index |
| :--- | :--- | :--- | :--- |
| **2 (Optimal)** | **93.64** | **0.6001** | **0.5421** |
| 3 | 51.58 | 0.5356 | 0.5567 |
| 4 | 31.34 | 0.5408 | 0.5441 |
| 5 | 20.92 | 0.5502 | 0.5201 |
| 6 | 15.40 | 0.5519 | 0.5207 |
| 7 | 13.44 | 0.5203 | 0.5646 |
| 8 | 10.82 | 0.5208 | 0.5424 |
| 9 | 9.96 | 0.5301 | 0.5139 |
| 10 | 8.61 | 0.5268 | 0.5009 |

Berdasarkan prioritas evaluasi, Silhouette Score tertinggi berada pada $k=2$ dengan nilai **0.6001** (yang berarti kualitas pemisahan kluster tergolong **BAIK**). Nilai Davies-Bouldin Index pada $k=2$ juga sangat rendah (**0.5421**), yang memvalidasi bahwa jarak antar kluster cukup lebar dan sebaran data di dalam kluster tergolong rapat.

#### **4.8 Analisis Hasil Clustering**
Kluster optimal $k=2$ membagi data pasien menjadi dua kelompok berdasarkan karakteristik usia:
* **Cluster 0 (Kelompok Usia Muda/Lebih Rendah)**:
  * Jumlah Anggota: 162 data (53.47%)
  * Rata-rata Usia Ternormalisasi: `-0.7755`
  * Rata-rata Usia Aktual: **$\approx 50.84$ tahun**
* **Cluster 1 (Kelompok Usia Tua/Lebih Tinggi)**:
  * Jumlah Anggota: 141 data (46.53%)
  * Rata-rata Usia Ternormalisasi: `0.8910`
  * Rata-rata Usia Aktual: **$\approx 68.16$ tahun**

Pembagian kluster K-Means manual berhasil menyegmentasikan pasien CAD Alizadeh secara seimbang (~53% usia muda dan ~46% usia tua) dengan batas pemisah usia yang jelas.

---

### **BAB V: KESIMPULAN DAN SARAN**

#### **5.1 Kesimpulan**
1. Metode klasifikasi supervised berhasil diterapkan pada dataset CAD Alizadeh. Algoritma **Random Forest** menunjukkan kinerja terbaik dengan akurasi **81.58%**, disusul oleh **MLP (78.95%)**, dan terakhir **Naive Bayes (53.95%)** yang kinerjanya buruk akibat pelanggaran asumsi independensi fitur.
2. Implementasi algoritma K-Means secara manual (tanpa library scikit-learn) berhasil dijalankan secara optimal untuk mengelompokkan data berdasarkan fitur usia (`Age`).
3. Berdasarkan evaluasi clustering menggunakan Silhouette Score, Elbow Method (SSE), dan Davies-Bouldin Index, jumlah kluster terbaik yang terbentuk adalah **$k=2$** dengan Silhouette Score **0.6001** (kategori **BAIK**). Kluster ini sukses membagi pasien menjadi kelompok usia muda (rata-rata 50.84 tahun) dan kelompok usia tua (rata-rata 68.16 tahun).

#### **5.2 Saran**
1. Pada penelitian klasifikasi supervised selanjutnya, dapat dicoba perluasan eksplorasi tuning hyperparameter (seperti menggunakan `GridSearchCV` secara riil pada Random Forest) untuk mengoptimasi *recall* pada kelas minoritas (`Normal`).
2. Proses clustering di masa mendatang sebaiknya mencoba menggabungkan beberapa fitur klinis prediktif sekaligus (multivariat) daripada hanya menggunakan satu fitur (`Age`) agar dapat memetakan kluster pasien secara lebih komprehensif.
3. Menangani ketidakseimbangan data prediktor (*data imbalance*) menggunakan teknik seperti SMOTE (*Synthetic Minority Over-sampling Technique*) agar performa *recall* kelas minoritas dapat ditingkatkan pada model klasifikasi.
