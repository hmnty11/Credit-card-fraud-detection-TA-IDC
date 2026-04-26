# Introduction to Data Science : TBHA - LEC
## Team Assignment 1 & 2 – Group 6

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
- Jumlah transaksi awal: 284.807, dengan 492 transaksi fraud (0,172%).
- Fitur: `Time`, `V1` … `V28` (hasil transformasi PCA), `Amount`, dan `Class` (label: 0 = normal, 1 = fraud).
- **Keunggulan dataset:**
  - Tidak memiliki *missing values*.
  - Didominasi fitur numerik (hasil PCA) sehingga mudah diolah.
  - Representasi *imbalanced class* yang menantang, sesuai dengan kondisi nyata.

---

## 🔍 Exploratory Data Analysis (EDA)

### Statistik Deskriptif
Kami menghitung statistik dasar untuk fitur numerik menggunakan `df.describe()` dan fungsi skewness/kurtosis dari `scipy.stats`.

| Fitur | Mean | Std | Skewness | Kurtosis |
|-------|------|-----|----------|----------|
| Time | 94.813,86 | 47.488,15 | -0,12 | -1,21 |
| Amount | 88,35 | 250,12 | 16,98 | 498,52 |
| V1..V28 | ~0 | ~1 | ~0 | ~0 |

- **Skewness Amount yang sangat tinggi** menunjukkan distribusi menceng kanan (banyak transaksi kecil, beberapa sangat besar).
- **Kurtosis Amount** sangat tinggi → banyak outlier ekstrem.

### Korelasi Antar Fitur
- Matriks korelasi divisualisasikan dengan **heatmap**.
- Fitur PCA (`V1`..`V28`) memiliki korelasi mendekati nol satu sama lain (sesuai sifat PCA).
- Korelasi `Amount` dengan `Class` sangat rendah (0,02), sedangkan beberapa fitur PCA memiliki korelasi lemah hingga sedang dengan `Class` (misal V11, V12, V14).

### Identifikasi Outlier
- **Metode IQR** pada `Amount` mengidentifikasi transaksi dengan nilai di luar batas `Q1 - 1.5*IQR` dan `Q3 + 1.5*IQR`.
- Sebanyak **31.685 outlier** dihapus dari kolom `Amount` (setelah duplikat dihapus).
- Outlier pada `Time` tidak ditangani karena tidak signifikan terhadap model.

### Visualisasi
- **Histogram Amount** menunjukkan distribusi miring dengan ekor panjang.
- **Countplot Class** memperlihatkan ketidakseimbangan kelas yang ekstrem.
- **Scatter plot V1 vs V2** menunjukkan bahwa titik fraud (merah) cenderung mengelompok di area tertentu, namun masih bercampur dengan normal.
- **Heatmap korelasi** menegaskan bahwa fitur PCA tidak berkorelasi.

### Interpretasi EDA
- **Pola utama:** Data sangat tidak seimbang; fitur PCA sudah baik untuk modelling; `Amount` perlu ditangani outlier dan diskalakan.
- **Outlier signifikan:** Ya, pada `Amount`. Telah dihapus menggunakan IQR.
- **Korelasi:** Fitur PCA tidak berkorelasi satu sama lain; beberapa memiliki korelasi kecil dengan `Class`. Tidak ada fitur yang redundan.

---

## ⚙️ Preprocessing & Feature Engineering

### A. Pembersihan Data
1. **Duplikat:** Dihapus (1.081 baris).
2. **Outlier Amount:** Dihapus menggunakan metode IQR (31.685 baris).
   - `df` akhir setelah cleaning: **252.041 baris**.

### B. Transformasi Data

#### Encoding Kategorikal
Dataset tidak memiliki fitur kategorikal (semua numerik). Tidak diperlukan encoding.

#### Standardisasi (Normalisasi)
Karena `Amount` dan `Time` memiliki skala berbeda dengan fitur PCA, kami melakukan **StandardScaler**:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
df['Time_scaled'] = scaler.fit_transform(df[['Time']])
df = df.drop(['Amount', 'Time'], axis=1)
```
- Tujuan: Agar semua fitur memiliki mean 0 dan std 1, sehingga model berbasis jarak (SVM, KNN) tidak bias.

#### Feature Engineering (Pembuatan Fitur Baru)
- Dibuat fitur `Amount_abs` = nilai absolut dari `Amount_scaled` untuk menangkap transaksi dengan nilai ekstrem (baik positif maupun negatif).

### C. Feature Selection & Extraction

#### PCA (Principal Component Analysis)
- Kami menerapkan PCA pada seluruh fitur (setelah scaling) untuk melihat reduksi dimensi.
- **Hasil:** 90% varians dapat dijelaskan oleh **10 komponen pertama**, 95% oleh **12 komponen**.
- Visualisasi 2D (PC1 vs PC2) menunjukkan pemisahan yang tidak sempurna antara fraud dan normal.

#### Feature Importance dengan Random Forest
- Karena data sangat tidak seimbang, kami mengambil sampel seimbang (fraud 1:5 normal).
- Random Forest (100 trees) menghitung importance setiap fitur.
- **10 fitur teratas:** V14, V12, V10, V17, V11, V4, V9, V16, V3, V7.
- Fitur-fitur ini dapat digunakan sebagai kandidat untuk model akhir.

### D. Output Akhir
Setelah feature engineering, dataset memiliki:
- **252.041 baris** dan **32 kolom** (30 fitur asli + `Amount_scaled` + `Time_scaled` + `Amount_abs`).
- Siap untuk pemodelan klasifikasi (dengan penanganan imbalance class seperti SMOTE).

---

## 📈 Kesimpulan dan Langkah Selanjutnya

| Tahap | Status |
|-------|--------|
| EDA lengkap | ✅ Selesai |
| Preprocessing (missing, duplikat, outlier) | ✅ Selesai |
| Standardisasi | ✅ Selesai |
| Feature Engineering (Amount_abs, PCA, Feature Importance) | ✅ Selesai |
| Dataset siap untuk modeling | ✅ |

**Langkah berikutnya:**
- Membangun model klasifikasi (Logistic Regression, Random Forest, XGBoost).
- Menangani ketidakseimbangan kelas dengan SMOTE atau class weights.
- Evaluasi dengan precision, recall, F1-score, AUC-ROC.

---

## 📁 Daftar File dalam Repository

- `CC Fraud Detection.ipynb` – Preprocessing awal, EDA, visualisasi.
- `CC Fraud Detection - Transformation and Feature Selection.ipynb` – Standardisasi, PCA, Feature Importance, pembuatan fitur baru.
- `creditcard.csv` – Dataset asli (dengan Git LFS).
- `requirements.txt` – Daftar library Python.
- `README.md` – Dokumentasi proyek.
- `LICENSE` – Lisensi MIT.

---

## 🔗 Cara Menjalankan

1. Clone repository ini.
2. Install dependencies: `pip install -r requirements.txt`
3. Buka notebook `.ipynb` menggunakan Jupyter atau VS Code.
4. Jalankan semua cell secara berurutan.

---

**Dibuat untuk memenuhi tugas**  
*Introduction to Data Science – COSC6028*
