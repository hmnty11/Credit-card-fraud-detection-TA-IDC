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
| Pengumpulan data | Kaggle API / unduh CSV |
| Eksplorasi data | Pandas, Matplotlib, Seaborn |
| Preprocessing | Scikit-learn (StandardScaler, SMOTE) |
| Pemodelan | Scikit-learn, XGBoost |
| Evaluasi | Classification report, confusion matrix, ROC curve |
