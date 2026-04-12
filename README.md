# Introduction to Data Science : TBHA - LEC
## Team Assignment 1 – Group 6

### Anggota Kelompok

| Nama | NIM |
|------|-----|
| BRANDON RAPHAEL VALENTINO | 2902735635 |
| DARREN JEHONATHAN | 2902714100 |
| MUHAMMAD DHIYAUL HAQ | 2902736934 |
| MUHAMMAD FATHIR | 2902730432 |
| SIMONE RAPHAEL ERUS | 2902703394 |

---

## 📌 Topik: Deteksi Transaksi Penipuan (Fraud) pada Kartu Kredit

### 1. Latar Belakang
Penipuan kartu kredit merupakan permasalahan nyata di industri keuangan yang dapat merugikan nasabah maupun institusi perbankan. Oleh karena itu, diperlukan sistem yang mampu mendeteksi transaksi mencurigakan secara otomatis. Dengan pendekatan **machine learning**, kita dapat membangun model klasifikasi untuk membedakan transaksi **normal** dan **fraud** (penipuan).

### 2. Dataset yang Digunakan
**Nama dataset:** Credit Card Fraud Detection  
**Sumber:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Deskripsi singkat:**
- Berisi transaksi kartu kredit oleh pemegang kartu Eropa pada September 2013.
- Jumlah transaksi: 284.807, dengan 492 transaksi fraud (0,172%).
- Fitur: `Time`, `V1` … `V28` (hasil transformasi PCA), `Amount`, dan `Class` (label: 0 = normal, 1 = fraud).
- **Keunggulan dataset:**
  - Tidak memiliki *missing values*.
  - Didominasi fitur numerik (hasil PCA) sehingga mudah diolah.
  - Representasi *imbalanced class* yang menantang, sesuai dengan kondisi nyata.

### 3. Relevansi dengan Data Science
- **Masalah:** Klasifikasi biner (normal vs fraud).
- **Tahapan yang akan dilakukan:**
  1. Eksplorasi data (EDA) – memahami distribusi, cek ketidakseimbangan kelas.
  2. Preprocessing – normalisasi `Amount` dan `Time`, mengatasi *class imbalance* (misal dengan SMOTE atau undersampling).
  3. Pemodelan – menggunakan algoritma seperti Logistic Regression, Random Forest, atau XGBoost.
  4. Evaluasi – menggunakan metrik seperti *precision*, *recall*, *F1-score*, dan *AUC-ROC* (karena akurasi tidak sesuai untuk data tidak seimbang).
- **Tujuan akhir:** Menghasilkan model yang dapat memprediksi transaksi fraud dengan tingkat deteksi tinggi dan *false positive* rendah.

### 4. Rencana Pengembangan
| Tahap | Metode/Alat |
|-------|--------------|
| Pengumpulan data | unduh CSV |
| Eksplorasi data | Pandas, Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (StandardScaler, SMOTE) |
| Pemodelan | Scikit-learn, XGBoost |
| Evaluasi | Classification report, confusion matrix, ROC curve |


### 5. Eksplorasi Awal
Dataset dibaca menggunakan pandas.

```python
import pandas as pd
df = pd.read_csv('creditcard.csv')
print(df.shape)          # (284807, 31)
print(df.info())
```
Hasil:
- Jumlah baris: 284.807
- Jumlah kolom: 31 (30 fitur numerik + 1 target Class)
- Tidak ada missing values pada kolom numerik.
- Kolom Time dan Amount memiliki rentang nilai yang sangat berbeda dengan fitur PCA (V1–V28).

### 6. Pembersihan Data 
**Identifikasi Masalah**

| Masalah | Ada? | Keterangan |
|---------|------|-------------|
| Missing values | Tidak | Semua kolom memiliki 284.807 non-null. |
| Outliers ekstrem | Ya | Pada kolom Amount dan Time. |
| Duplikasi data | Ya | Terdapat 1.081 baris duplikat. |

**Penanganan Duplikasi**
Kami menghapus baris duplikat menggunakan drop_duplicates().

```python
df = df.drop_duplicates()
print(df.shape)  # (283726, 31)
```
Alasan: Duplikat dapat menyebabkan bias dalam model, terutama pada data yang tidak seimbang (fraud sangat sedikit).

**Penanganan Outlier**
Outlier pada kolom Amount diidentifikasi menggunakan metode IQR (Interquartile Range).

```python
Q1 = df['Amount'].quantile(0.25)
Q3 = df['Amount'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df = df[(df['Amount'] >= lower) & (df['Amount'] <= upper)]
```

Outlier ekstrem pada Amount (misal transaksi hingga >25.000) dapat mengganggu normalisasi dan kinerja model. Kami memilih IQR karena distribusi Amount tidak simetris.
Setelah penghapusan outlier, jumlah baris menjadi 252.041.

### 7. Data Preprocessing 
**Standarisasi (Normalisasi)**
Karena Amount dan Time memiliki rentang yang sangat berbeda dengan fitur PCA, kami melakukan standarisasi menggunakan StandardScaler dari scikit-learn.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)
```
Standarisasi diperlukan agar semua fitur memiliki skala yang sama (mean=0, std=1), sehingga algoritma berbasis jarak (seperti SVM, KNN) tidak bias terhadap fitur dengan skala besar.

**Reduksi Dimensi**
Kami tidak melakukan reduksi dimensi karena jumlah fitur (30) masih tergolong kecil dan PCA sudah diterapkan pada data asli (fitur V1–V28 merupakan hasil PCA dari data asli).

**Visualisasi Sebelum dan Sesudah Preprocessing** 
Histogram Amount (Sebelum vs Sesudah Scaling)
Sebelum: Distribusi menceng ke kanan (skewed), banyak outlier.
Sesudah: Distribusi mendekati normal (mean 0, std 1).

**Perbandingan Jumlah Fraud vs Normal**
Tetap sangat tidak seimbang (fraud ~0.17%). Visualisasi dengan countplot menunjukkan dominasi kelas normal.

**Heatmap Korelasi**
Korelasi antar fitur PCA sangat rendah (mendekati 0), sesuai dengan sifat PCA yang menghasilkan komponen tidak berkorelasi.

**Scatter Plot (V1 vs V2)**
Titik merah (fraud) cenderung berada di area tertentu, namun masih bercampur dengan normal. Menunjukkan perlunya model yang lebih kompleks.

### Kesimpulan setelah Preprocessing
| Pertanyaan | Jawaban |
|------------|---------|
| Apakah dataset sudah cukup bersih untuk analisis? | Ya. Missing values tidak ada, duplikat dihapus, outlier pada Amount telah ditangani, dan fitur telah distandarisasi. |
| Apakah masih ada masalah? | Ketidakseimbangan kelas (fraud sangat sedikit) masih menjadi tantangan utama. Perlu teknik resampling (SMOTE, undersampling) pada tahap pemodelan. |
| Tantangan utama dalam proses ini | Menentukan batas IQR untuk outlier tanpa menghilangkan transaksi fraud yang sah (tidak ada fraud yang terhapus karena semua fraud memiliki Amount relatif kecil). |

Dataset siap untuk dilanjutkan ke tahap pemodelan machine learning (klasifikasi biner) dengan menangani ketidakseimbangan kelas terlebih dahulu.
