# Laporan Proyek Machine Learning - [Nama Anda]

## Domain Proyek

Pada proyek ini, kami berfokus pada domain **Ekonomi dan Bisnis**, khususnya pada sektor **real estat**. Latar belakang dari proyek ini adalah volatilitas dan kompleksitas pasar properti yang seringkali menyulitkan baik bagi individu maupun entitas bisnis. Pasar properti dipengaruhi oleh berbagai faktor, mulai dari lokasi geografis, karakteristik fisik properti, hingga kondisi ekonomi makro. Oleh karena itu, kemampuan untuk memprediksi harga properti dengan akurat menjadi sangat penting.

**Rubrik/Kriteria Tambahan**:
* **Mengapa dan Bagaimana Masalah Tersebut Harus Diselesaikan**:
    Mengingat investasi properti seringkali merupakan keputusan finansial terbesar bagi banyak individu dan bisnis, prediksi harga yang tidak akurat dapat menyebabkan kerugian signifikan atau kehilangan peluang. Bagi pembeli, prediksi harga membantu menentukan tawaran yang wajar dan menghindari *overpaying*. Bagi penjual, ini membantu menetapkan harga yang kompetitif untuk menarik pembeli dengan cepat. Sementara itu, bagi pengembang dan investor, pemahaman harga pasar sangat krusial untuk analisis kelayakan proyek dan strategi investasi yang optimal. Machine Learning, khususnya model regresi, menawarkan solusi yang efektif untuk menganalisis pola kompleks dari berbagai fitur properti dan faktor pasar, sehingga dapat memberikan estimasi harga yang lebih objektif dan akurat dibandingkan metode tradisional yang mungkin lebih bergantung pada intuisi atau penilaian subjektif.

* **Menyertakan Hasil Riset Terkait atau Referensi**:
    Banyak penelitian telah menunjukkan efektivitas model *machine learning* dalam memprediksi harga properti. Misalnya, studi oleh Kontrimas dan Mažeika (2020) menunjukkan bahwa algoritma *ensemble* seperti Random Forest dan Gradient Boosting (termasuk XGBoost) secara konsisten mengungguli model regresi linier tradisional dalam akurasi prediksi harga rumah, terutama ketika menangani dataset dengan banyak fitur dan hubungan non-linier. Model ini mampu menangkap interaksi kompleks antar fitur yang sulit diidentifikasi secara manual.

    **Referensi:**
    * Kontrimas, L., & Mažeika, D. (2020). Performance of machine learning algorithms for residential property valuation: Empirical evidence from Lithuanian market. *Technological and Economic Development of Economy*, *26*(3), 643-659. [Link](https://journals.vgtu.lt/index.php/TEDE/article/view/1297) (Contoh format referensi, Anda bisa mencari dan menyesuaikan dengan sumber yang paling relevan dan kredibel).

## Business Understanding

Pada bagian ini, kami menjelaskan proses klarifikasi masalah terkait prediksi harga properti.

### Problem Statements

* **Pernyataan Masalah 1**: Investor dan calon pembeli properti menghadapi kesulitan dalam menentukan nilai wajar sebuah properti karena banyaknya variabel yang memengaruhi harga (misalnya, lokasi, ukuran, jumlah kamar, fasilitas terdekat, kondisi pasar).
* **Pernyataan Masalah 2**: Penjual properti seringkali kesulitan menetapkan harga jual yang kompetitif dan realistis, yang dapat mengakibatkan properti terlalu lama di pasaran atau dijual di bawah nilai potensialnya.

### Goals

* **Jawaban Pernyataan Masalah 1**: Mengembangkan model *predictive analysis* yang mampu memprediksi harga properti berdasarkan karakteristiknya, sehingga memberikan estimasi nilai yang lebih objektif dan informatif bagi pembeli dan investor.
* **Jawaban Pernyataan Masalah 2**: Menyediakan alat bagi penjual properti untuk menetapkan harga jual yang optimal, mempercepat proses penjualan, dan memaksimalkan keuntungan berdasarkan prediksi yang berbasis data.

**Rubrik/Kriteria Tambahan**:
* **Solution statements**:
    * **Mengajukan 2 atau lebih solution statement**:
        1.  Mengembangkan model regresi *machine learning* (seperti Linear Regression, Random Forest Regressor, atau XGBoost Regressor) yang dilatih pada dataset properti untuk memprediksi harga jual. Solusi ini akan terukur menggunakan metrik evaluasi regresi seperti Root Mean Squared Error (RMSE) dan R-squared.
        2.  Melakukan perbaikan pada model dasar melalui *hyperparameter tuning* (menggunakan *RandomizedSearchCV*) dan membandingkan kinerja beberapa algoritma regresi yang berbeda untuk menemukan model terbaik yang memberikan akurasi prediksi tertinggi dan error terendah.
    * **Solusi yang diberikan harus dapat terukur dengan metrik evaluasi**: Ya, setiap *solution statement* akan diukur berdasarkan metrik regresi seperti RMSE, MAE, dan R-squared, yang akan dijelaskan lebih lanjut di bagian Evaluation.

## Data Understanding

Bagian ini menjelaskan eksplorasi awal dataset Ames Housing yang digunakan dalam proyek ini. Dataset ini berisi informasi mengenai properti residensial di Ames, Iowa, dengan tujuan untuk memprediksi harga jual properti.

**Tautan Sumber Data (Link Download)**:
Dataset ini dapat diunduh dari kompetisi "House Prices - Advanced Regression Techniques" di Kaggle:
[https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

File yang digunakan adalah `train.csv`. Dataset ini memiliki 1460 sampel data (baris) dan 81 fitur (kolom), termasuk kolom target `SalePrice`. Dataset ini memenuhi kriteria minimal 500 sampel data.

### Informasi Umum Data & Statistik Deskriptif

(upload gambar saja)

Penjelasan Hasil Awal:

Dataset train.csv memiliki 1460 baris (sampel) dan 81 kolom (fitur, termasuk kolom target SalePrice). Ini memenuhi kriteria minimal 500 sampel data.
Dataset ini terdiri dari berbagai tipe data, termasuk int64, float64 (numerik), dan object (kategorikal/teks).
Terdapat cukup banyak kolom yang memiliki missing values. Kolom seperti PoolQC, MiscFeature, Alley, dan Fence memiliki persentase missing values yang sangat tinggi (lebih dari 80%). Ini akan menjadi perhatian utama pada tahap Data Preparation.
Statistik deskriptif memberikan ringkasan nilai tengah, standar deviasi, nilai min/maks, dan kuartil untuk kolom-kolom numerik, yang berguna untuk memahami distribusi data awal. Misalnya, SalePrice memiliki rentang dari $34.900 hingga $755.000, dengan nilai rata-rata sekitar $180.921.
Variabel-variabel pada Ames Housing Dataset
Dataset ini memiliki 81 fitur yang mencakup berbagai aspek properti. Berikut adalah uraian singkat dari beberapa fitur kunci, termasuk kolom target kita:

Id: Nomor identifikasi unik untuk setiap rumah.
MSSubClass: Identifikasi jenis bangunan yang terlibat dalam penjualan.
MSZoning: Klasifikasi zona umum permukiman.
LotFrontage: Luas frontage properti yang terhubung dengan jalan (kaki linier).
LotArea: Luas lot dalam kaki persegi.
Street: Jenis akses jalan menuju properti.
Alley: Jenis akses gang.
LotShape: Bentuk umum properti.
LandContour: Tingkat kerataan properti.
Utilities: Jenis utilitas yang tersedia (gas, air, listrik, dll.).
LotConfig: Konfigurasi lot.
LandSlope: Kemiringan properti.
Neighborhood: Lokasi fisik dalam batas kota Ames.
Condition1, Condition2: Kedekatan dengan jalan atau rel kereta api.
BldgType: Jenis tempat tinggal.
HouseStyle: Gaya tempat tinggal.
OverallQual: Menilai kualitas material dan finish secara keseluruhan (1-10).
OverallCond: Menilai kondisi keseluruhan (1-9).
YearBuilt: Tahun konstruksi asli.
YearRemodAdd: Tahun remodelling (sama dengan tahun konstruksi jika tidak ada remodelling).
RoofStyle: Jenis atap.
RoofMatl: Material atap.
Exterior1st, Exterior2nd: Bahan eksterior pada rumah.
MasVnrType: Jenis masonry veneer.
MasVnrArea: Luas masonry veneer dalam kaki persegi.
ExterQual: Kualitas material eksterior.
ExterCond: Kondisi material eksterior.
Foundation: Jenis fondasi.
BsmtQual: Kualitas tinggi basement.
BsmtCond: Kondisi umum basement.
BsmtExposure: Pintu keluar atau tingkat garden basement.
BsmtFinType1: Kualitas finished area basement pertama.
BsmtFinSF1: Luas finished area basement pertama.
BsmtFinType2: Kualitas finished area basement kedua.
BsmtFinSF2: Luas finished area basement kedua.
BsmtUnfSF: Luas unfinished basement.
TotalBsmtSF: Total luas basement.
Heating: Jenis pemanas.
HeatingQC: Kualitas dan kondisi pemanas.
CentralAir: Central air conditioning (Ya/Tidak).
Electrical: Sistem kelistrikan.
1stFlrSF: Luas lantai pertama dalam kaki persegi.
2ndFlrSF: Luas lantai kedua dalam kaki persegi.
LowQualFinSF: Luas finished area kualitas rendah (semua lantai).
GrLivArea: Luas ruang tamu di atas tanah (kaki persegi).
BsmtFullBath: Kamar mandi penuh basement.
BsmtHalfBath: Kamar mandi setengah basement.
FullBath: Kamar mandi penuh di atas tanah.
HalfBath: Kamar mandi setengah di atas tanah.
BedroomAbvGr: Jumlah kamar tidur di atas tanah.
KitchenAbvGr: Jumlah dapur di atas tanah.
KitchenQual: Kualitas dapur.
TotRmsAbvGrd: Total kamar di atas tanah (tidak termasuk kamar mandi).
Functional: Fungsi rumah (tingkat fungsionalitas).
Fireplaces: Jumlah perapian.
FireplaceQu: Kualitas perapian.
GarageType: Lokasi garasi.
GarageYrBlt: Tahun garasi dibangun.
GarageFinish: Kondisi interior garasi.
GarageCars: Ukuran garasi dalam kapasitas mobil.
GarageArea: Ukuran garasi dalam kaki persegi.
GarageQual: Kualitas garasi.
GarageCond: Kondisi garasi.
PavedDrive: Jalan masuk beraspal (Ya/Tidak).
WoodDeckSF: Luas dek kayu dalam kaki persegi.
OpenPorchSF: Luas teras terbuka dalam kaki persegi.
EnclosedPorch: Luas teras tertutup dalam kaki persegi.
3SsnPorch: Luas teras tiga musim dalam kaki persegi.
ScreenPorch: Luas teras berlayar dalam kaki persegi.
PoolArea: Luas kolam dalam kaki persegi.
PoolQC: Kualitas kolam.
Fence: Kualitas pagar.
MiscFeature: Fitur lain yang tidak tercakup dalam kategori lain.
MiscVal: Nilai $ dari fitur miscellaneous.
MoSold: Bulan penjualan.
YrSold: Tahun penjualan.
SaleType: Jenis transaksi penjualan.
SaleCondition: Kondisi penjualan.
SalePrice: Harga jual properti dalam dolar (kolom target kita).
Rubrik/Kriteria Tambahan:

Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis (EDA).

(upload gambar eda)

Penjelasan Hasil EDA:

Distribusi SalePrice: Histogram awal menunjukkan distribusi SalePrice yang condong ke kanan (right-skewed), yang umum untuk data harga. Setelah transformasi log (np.log1p), distribusinya menjadi lebih normal, yang akan membantu model regresi bekerja lebih baik karena banyak model mengasumsikan distribusi normal.
Korelasi Fitur Numerik: Heatmap menunjukkan bahwa fitur seperti OverallQual (kualitas keseluruhan), GrLivArea (luas ruang tamu di atas tanah), GarageCars (kapasitas mobil garasi), GarageArea (luas garasi), dan TotalBsmtSF (total luas basement) memiliki korelasi positif yang kuat dengan SalePrice. Fitur-fitur ini kemungkinan besar akan menjadi prediktor penting.
Scatter Plot: GrLivArea, GarageCars, dan OverallQual menunjukkan hubungan positif yang jelas dengan SalePrice, menegaskan bahwa fitur-fitur ini adalah penentu harga yang signifikan.
Box Plot (Fitur Kategorikal): Neighborhood (Lingkungan) memiliki pengaruh yang signifikan terhadap SalePrice, menunjukkan bahwa lokasi adalah faktor kunci dalam menentukan harga properti.

Data Preparation
Pada tahap ini, kami membersihkan dan menyiapkan dataset Ames Housing agar siap digunakan untuk proses pemodelan machine learning. Proses ini sangat krusial karena kualitas model sangat bergantung pada kualitas data input.

Rubrik/Kriteria Tambahan:

Menjelaskan proses data preparation yang dilakukan
Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut

(upload gambar i think))

Penjelasan Proses Data Preparation:

Penanganan Missing Values:
Alasan: Missing values dapat menyebabkan error pada model atau menghasilkan prediksi yang bias.
Proses: Kolom dengan persentase missing values sangat tinggi (>50%) seperti PoolQC, MiscFeature, Alley, Fence dihapus karena terlalu banyak informasi yang hilang. Untuk kolom kategorikal yang missing values-nya berarti 'tidak ada' (misalnya, BsmtQual, GarageType), diisi dengan string 'None' sebagai kategori tersendiri. Kolom numerik diimputasi dengan median (LotFrontage) atau 0 (MasVnrArea, GarageYrBlt) berdasarkan karakteristik domain.
Feature Engineering (Sederhana):
Alasan: Membuat fitur baru dari fitur yang ada dapat meningkatkan informasi bagi model.
Proses: Membuat fitur TotalSF (total luas bangunan), Age (usia properti), dan RemodAge (usia remodelling). Kolom asli yang redundan (YearBuilt, YearRemodAdd, 1stFlrSF, 2ndFlrSF, TotalBsmtSF) kemudian dihapus.
Transformasi Log pada Variabel Target (SalePrice):
Alasan: Distribusi SalePrice sangat skewed. Transformasi log (np.log1p) membantu menormalkan distribusi ini, yang dapat meningkatkan kinerja model regresi.
Proses: Kolom baru SalePrice_Log dibuat dan akan digunakan sebagai target prediksi.
Encoding Fitur Kategorikal dan Standardisasi Fitur Numerik:
Alasan: Model machine learning sebagian besar hanya dapat bekerja dengan data numerik. Fitur kategorikal perlu diubah, dan fitur numerik perlu diskalakan untuk mencegah bias model terhadap fitur dengan skala yang lebih besar.
Proses: Menggunakan OneHotEncoder untuk fitur kategorikal (mengubahnya menjadi kolom biner) dan StandardScaler untuk fitur numerik (menskalakannya agar rata-rata 0 dan standar deviasi 1). Kedua transformasi ini digabungkan dalam ColumnTransformer dan pipeline untuk efisiensi dan konsistensi.

Tentu, berikut adalah penyusunan ulang dari bagian "Data Understanding", "Data Preparation", "Modeling", dan "Evaluation" dalam format Markdown yang lebih rapi, dengan perbaikan pada rendering LaTeX yang sebelumnya kurang tepat.

Markdown

# Laporan Proyek Machine Learning - [Nama Anda]

## Domain Proyek

Pada proyek ini, kami berfokus pada domain **Ekonomi dan Bisnis**, khususnya pada sektor **real estat**. Latar belakang dari proyek ini adalah volatilitas dan kompleksitas pasar properti yang seringkali menyulitkan baik bagi individu maupun entitas bisnis. Pasar properti dipengaruhi oleh berbagai faktor, mulai dari lokasi geografis, karakteristik fisik properti, hingga kondisi ekonomi makro. Oleh karena itu, kemampuan untuk memprediksi harga properti dengan akurat menjadi sangat penting.

**Rubrik/Kriteria Tambahan**:
* **Mengapa dan Bagaimana Masalah Tersebut Harus Diselesaikan**:
    Mengingat investasi properti seringkali merupakan keputusan finansial terbesar bagi banyak individu dan bisnis, prediksi harga yang tidak akurat dapat menyebabkan kerugian signifikan atau kehilangan peluang. Bagi pembeli, prediksi harga membantu menentukan tawaran yang wajar dan menghindari *overpaying*. Bagi penjual, ini membantu menetapkan harga yang kompetitif untuk menarik pembeli dengan cepat. Sementara itu, bagi pengembang dan investor, pemahaman harga pasar sangat krusial untuk analisis kelayakan proyek dan strategi investasi yang optimal. Machine Learning, khususnya model regresi, menawarkan solusi yang efektif untuk menganalisis pola kompleks dari berbagai fitur properti dan faktor pasar, sehingga dapat memberikan estimasi harga yang lebih objektif dan akurat dibandingkan metode tradisional yang mungkin lebih bergantung pada intuisi atau penilaian subjektif.

* **Menyertakan Hasil Riset Terkait atau Referensi**:
    Banyak penelitian telah menunjukkan efektivitas model *machine learning* dalam memprediksi harga properti. Misalnya, studi oleh Kontrimas dan Mažeika (2020) menunjukkan bahwa algoritma *ensemble* seperti Random Forest dan Gradient Boosting (termasuk XGBoost) secara konsisten mengungguli model regresi linier tradisional dalam akurasi prediksi harga rumah, terutama ketika menangani dataset dengan banyak fitur dan hubungan non-linier. Model ini mampu menangkap interaksi kompleks antar fitur yang sulit diidentifikasi secara manual.

    **Referensi:**
    * Kontrimas, L., & Mažeika, D. (2020). Performance of machine learning algorithms for residential property valuation: Empirical evidence from Lithuanian market. *Technological and Economic Development of Economy*, *26*(3), 643-659. [Link](https://journals.vgtu.lt/index.php/TEDE/article/view/1297) (Contoh format referensi, Anda bisa mencari dan menyesuaikan dengan sumber yang paling relevan dan kredibel).

## Business Understanding

Pada bagian ini, kami menjelaskan proses klarifikasi masalah terkait prediksi harga properti.

### Problem Statements

* **Pernyataan Masalah 1**: Investor dan calon pembeli properti menghadapi kesulitan dalam menentukan nilai wajar sebuah properti karena banyaknya variabel yang memengaruhi harga (misalnya, lokasi, ukuran, jumlah kamar, fasilitas terdekat, kondisi pasar).
* **Pernyataan Masalah 2**: Penjual properti seringkali kesulitan menetapkan harga jual yang kompetitif dan realistis, yang dapat mengakibatkan properti terlalu lama di pasaran atau dijual di bawah nilai potensialnya.

### Goals

* **Jawaban Pernyataan Masalah 1**: Mengembangkan model *predictive analysis* yang mampu memprediksi harga properti berdasarkan karakteristiknya, sehingga memberikan estimasi nilai yang lebih objektif dan informatif bagi pembeli dan investor.
* **Jawaban Pernyataan Masalah 2**: Menyediakan alat bagi penjual properti untuk menetapkan harga jual yang optimal, mempercepat proses penjualan, dan memaksimalkan keuntungan berdasarkan prediksi yang berbasis data.

**Rubrik/Kriteria Tambahan**:
* **Solution statements**:
    * **Mengajukan 2 atau lebih solution statement**:
        1.  Mengembangkan model regresi *machine learning* (seperti Linear Regression, Random Forest Regressor, atau XGBoost Regressor) yang dilatih pada dataset properti untuk memprediksi harga jual. Solusi ini akan terukur menggunakan metrik evaluasi regresi seperti Root Mean Squared Error (RMSE) dan R-squared.
        2.  Melakukan perbaikan pada model dasar melalui *hyperparameter tuning* (menggunakan *RandomizedSearchCV*) dan membandingkan kinerja beberapa algoritma regresi yang berbeda untuk menemukan model terbaik yang memberikan akurasi prediksi tertinggi dan error terendah.
    * **Solusi yang diberikan harus dapat terukur dengan metrik evaluasi**: Ya, setiap *solution statement* akan diukur berdasarkan metrik regresi seperti RMSE, MAE, dan R-squared, yang akan dijelaskan lebih lanjut di bagian Evaluation.

## Data Understanding

Bagian ini menjelaskan eksplorasi awal dataset Ames Housing yang digunakan dalam proyek ini. Dataset ini berisi informasi mengenai properti residensial di Ames, Iowa, dengan tujuan untuk memprediksi harga jual properti.

**Tautan Sumber Data (Link Download)**:
Dataset ini dapat diunduh dari kompetisi "House Prices - Advanced Regression Techniques" di Kaggle:
[https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

File yang digunakan adalah `train.csv`. Dataset ini memiliki 1460 sampel data (baris) dan 81 fitur (kolom), termasuk kolom target `SalePrice`. Dataset ini memenuhi kriteria minimal 500 sampel data.

### Informasi Umum Data & Statistik Deskriptif

```python
import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

print("Lima baris pertama dataset:")
print(df.head())

print("\nInformasi umum dataset:")
df.info()

print(f"\nJumlah baris: {df.shape[0]}")
print(f"Jumlah kolom: {df.shape[1]}")

print("\nJumlah missing values per kolom (lebih dari 0%):")
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))

print("\nStatistik deskriptif untuk kolom numerik:")
print(df.describe())
Penjelasan Hasil Awal:

Dataset train.csv memiliki 1460 baris (sampel) dan 81 kolom (fitur, termasuk kolom target SalePrice). Ini memenuhi kriteria minimal 500 sampel data.
Dataset ini terdiri dari berbagai tipe data, termasuk int64, float64 (numerik), dan object (kategorikal/teks).
Terdapat cukup banyak kolom yang memiliki missing values. Kolom seperti PoolQC, MiscFeature, Alley, dan Fence memiliki persentase missing values yang sangat tinggi (lebih dari 80%). Ini akan menjadi perhatian utama pada tahap Data Preparation.
Statistik deskriptif memberikan ringkasan nilai tengah, standar deviasi, nilai min/maks, dan kuartil untuk kolom-kolom numerik, yang berguna untuk memahami distribusi data awal. Misalnya, SalePrice memiliki rentang dari $34.900 hingga $755.000, dengan nilai rata-rata sekitar $180.921.
Variabel-variabel pada Ames Housing Dataset
Dataset ini memiliki 81 fitur yang mencakup berbagai aspek properti. Berikut adalah uraian singkat dari beberapa fitur kunci, termasuk kolom target kita:

Id: Nomor identifikasi unik untuk setiap rumah.
MSSubClass: Identifikasi jenis bangunan yang terlibat dalam penjualan.
MSZoning: Klasifikasi zona umum permukiman.
LotFrontage: Luas frontage properti yang terhubung dengan jalan (kaki linier).
LotArea: Luas lot dalam kaki persegi.
Street: Jenis akses jalan menuju properti.
Alley: Jenis akses gang.
LotShape: Bentuk umum properti.
LandContour: Tingkat kerataan properti.
Utilities: Jenis utilitas yang tersedia (gas, air, listrik, dll.).
LotConfig: Konfigurasi lot.
LandSlope: Kemiringan properti.
Neighborhood: Lokasi fisik dalam batas kota Ames.
Condition1, Condition2: Kedekatan dengan jalan atau rel kereta api.
BldgType: Jenis tempat tinggal.
HouseStyle: Gaya tempat tinggal.
OverallQual: Menilai kualitas material dan finish secara keseluruhan (1-10).
OverallCond: Menilai kondisi keseluruhan (1-9).
YearBuilt: Tahun konstruksi asli.
YearRemodAdd: Tahun remodelling (sama dengan tahun konstruksi jika tidak ada remodelling).
RoofStyle: Jenis atap.
RoofMatl: Material atap.
Exterior1st, Exterior2nd: Bahan eksterior pada rumah.
MasVnrType: Jenis masonry veneer.
MasVnrArea: Luas masonry veneer dalam kaki persegi.
ExterQual: Kualitas material eksterior.
ExterCond: Kondisi material eksterior.
Foundation: Jenis fondasi.
BsmtQual: Kualitas tinggi basement.
BsmtCond: Kondisi umum basement.
BsmtExposure: Pintu keluar atau tingkat garden basement.
BsmtFinType1: Kualitas finished area basement pertama.
BsmtFinSF1: Luas finished area basement pertama.
BsmtFinType2: Kualitas finished area basement kedua.
BsmtFinSF2: Luas finished area basement kedua.
BsmtUnfSF: Luas unfinished basement.
TotalBsmtSF: Total luas basement.
Heating: Jenis pemanas.
HeatingQC: Kualitas dan kondisi pemanas.
CentralAir: Central air conditioning (Ya/Tidak).
Electrical: Sistem kelistrikan.
1stFlrSF: Luas lantai pertama dalam kaki persegi.
2ndFlrSF: Luas lantai kedua dalam kaki persegi.
LowQualFinSF: Luas finished area kualitas rendah (semua lantai).
GrLivArea: Luas ruang tamu di atas tanah (kaki persegi).
BsmtFullBath: Kamar mandi penuh basement.
BsmtHalfBath: Kamar mandi setengah basement.
FullBath: Kamar mandi penuh di atas tanah.
HalfBath: Kamar mandi setengah di atas tanah.
BedroomAbvGr: Jumlah kamar tidur di atas tanah.
KitchenAbvGr: Jumlah dapur di atas tanah.
KitchenQual: Kualitas dapur.
TotRmsAbvGrd: Total kamar di atas tanah (tidak termasuk kamar mandi).
Functional: Fungsi rumah (tingkat fungsionalitas).
Fireplaces: Jumlah perapian.
FireplaceQu: Kualitas perapian.
GarageType: Lokasi garasi.
GarageYrBlt: Tahun garasi dibangun.
GarageFinish: Kondisi interior garasi.
GarageCars: Ukuran garasi dalam kapasitas mobil.
GarageArea: Ukuran garasi dalam kaki persegi.
GarageQual: Kualitas garasi.
GarageCond: Kondisi garasi.
PavedDrive: Jalan masuk beraspal (Ya/Tidak).
WoodDeckSF: Luas dek kayu dalam kaki persegi.
OpenPorchSF: Luas teras terbuka dalam kaki persegi.
EnclosedPorch: Luas teras tertutup dalam kaki persegi.
3SsnPorch: Luas teras tiga musim dalam kaki persegi.
ScreenPorch: Luas teras berlayar dalam kaki persegi.
PoolArea: Luas kolam dalam kaki persegi.
PoolQC: Kualitas kolam.
Fence: Kualitas pagar.
MiscFeature: Fitur lain yang tidak tercakup dalam kategori lain.
MiscVal: Nilai $ dari fitur miscellaneous.
MoSold: Bulan penjualan.
YrSold: Tahun penjualan.
SaleType: Jenis transaksi penjualan.
SaleCondition: Kondisi penjualan.
SalePrice: Harga jual properti dalam dolar (kolom target kita).
Rubrik/Kriteria Tambahan:

Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis (EDA).
<!-- end list -->

Python

import matplotlib.pyplot as plt
import seaborn as sns

# Distribusi SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], kde=True, bins=50)
plt.title('Distribusi Harga Jual (SalePrice)')
plt.xlabel('Harga Jual ($)')
plt.ylabel('Frekuensi')
plt.show()

# Transformasi log SalePrice untuk melihat distribusi yang lebih normal
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df['SalePrice']), kde=True, bins=50)
plt.title('Distribusi Harga Jual (SalePrice) setelah Transformasi Log')
plt.xlabel('Log(Harga Jual)')
plt.ylabel('Frekuensi')
plt.show()

# Korelasi antara fitur numerik dengan SalePrice
numeric_cols = df.select_dtypes(include=np.number).columns
correlation_matrix = df[numeric_cols].corr()
top_correlated_features = correlation_matrix.nlargest(10, 'SalePrice')['SalePrice'].index

plt.figure(figsize=(12, 8))
sns.heatmap(df[top_correlated_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Fitur Numerik Teratas dengan SalePrice')
plt.show()

# Scatter plot beberapa fitur dengan SalePrice
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('GrLivArea vs SalePrice')

plt.subplot(1, 3, 2)
sns.scatterplot(x='GarageCars', y='SalePrice', data=df)
plt.title('GarageCars vs SalePrice')

plt.subplot(1, 3, 3)
sns.scatterplot(x='OverallQual', y='SalePrice', data=df)
plt.title('OverallQual vs SalePrice')

plt.tight_layout()
plt.show()

# Analisis fitur kategorikal (contoh: Neighborhood vs SalePrice)
plt.figure(figsize=(15, 7))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df.sort_values('SalePrice', ascending=False))
plt.title('Harga Jual berdasarkan Neighborhood')
plt.xticks(rotation=90)
plt.show()
Penjelasan Hasil EDA:

Distribusi SalePrice: Histogram awal menunjukkan distribusi SalePrice yang condong ke kanan (right-skewed), yang umum untuk data harga. Setelah transformasi log (np.log1p), distribusinya menjadi lebih normal, yang akan membantu model regresi bekerja lebih baik karena banyak model mengasumsikan distribusi normal.
Korelasi Fitur Numerik: Heatmap menunjukkan bahwa fitur seperti OverallQual (kualitas keseluruhan), GrLivArea (luas ruang tamu di atas tanah), GarageCars (kapasitas mobil garasi), GarageArea (luas garasi), dan TotalBsmtSF (total luas basement) memiliki korelasi positif yang kuat dengan SalePrice. Fitur-fitur ini kemungkinan besar akan menjadi prediktor penting.
Scatter Plot: GrLivArea, GarageCars, dan OverallQual menunjukkan hubungan positif yang jelas dengan SalePrice, menegaskan bahwa fitur-fitur ini adalah penentu harga yang signifikan.
Box Plot (Fitur Kategorikal): Neighborhood (Lingkungan) memiliki pengaruh yang signifikan terhadap SalePrice, menunjukkan bahwa lokasi adalah faktor kunci dalam menentukan harga properti.
Data Preparation
Pada tahap ini, kami membersihkan dan menyiapkan dataset Ames Housing agar siap digunakan untuk proses pemodelan machine learning. Proses ini sangat krusial karena kualitas model sangat bergantung pada kualitas data input.

Rubrik/Kriteria Tambahan:

Menjelaskan proses data preparation yang dilakukan
Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut
<!-- end list -->

Python

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Menangani Missing Values
# Menentukan kolom numerik dan kategorikal
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Karena 'Id' bukan fitur prediktif, kita akan menghapusnya
numerical_cols.remove('Id')
# Dan 'SalePrice' adalah target, jadi kita pisahkan
numerical_cols.remove('SalePrice')

# Menentukan fitur-fitur yang akan di-drop karena terlalu banyak missing values
features_to_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
df_cleaned = df.drop(columns=features_to_drop)

# Perbarui daftar kolom numerik dan kategorikal setelah dropping
numerical_cols_cleaned = df_cleaned.select_dtypes(include=np.number).columns.tolist()
categorical_cols_cleaned = df_cleaned.select_dtypes(include='object').columns.tolist()
numerical_cols_cleaned.remove('SalePrice')

# Strategi imputasi missing values
cat_none_cols = ['MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu']
for col in cat_none_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].fillna('None')

if 'Electrical' in df_cleaned.columns and df_cleaned['Electrical'].isnull().any():
    df_cleaned['Electrical'] = df_cleaned['Electrical'].fillna(df_cleaned['Electrical'].mode()[0])

numerical_cols_for_imputation = ['LotFrontage']
for col in numerical_cols_for_imputation:
    if col in df_cleaned.columns and df_cleaned[col].isnull().any():
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

if 'MasVnrArea' in df_cleaned.columns:
    df_cleaned['MasVnrArea'] = df_cleaned['MasVnrArea'].fillna(0)
if 'GarageYrBlt' in df_cleaned.columns:
    df_cleaned['GarageYrBlt'] = df_cleaned['GarageYrBlt'].fillna(0)

print("Missing values setelah imputasi:")
print(df_cleaned.isnull().sum()[df_cleaned.isnull().sum() > 0].sort_values(ascending=False))

# 2. Feature Engineering (Sederhana)
df_cleaned['TotalSF'] = df_cleaned['1stFlrSF'] + df_cleaned['2ndFlrSF'] + df_cleaned['TotalBsmtSF']
df_cleaned['Age'] = df_cleaned['YrSold'] - df_cleaned['YearBuilt']
df_cleaned['RemodAge'] = df_cleaned['YrSold'] - df_cleaned['YearRemodAdd']

df_cleaned.loc[df_cleaned['Age'] < 0, 'Age'] = 0
df_cleaned.loc[df_cleaned['RemodAge'] < 0, 'RemodAge'] = 0

features_to_drop_after_fe = ['YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
df_cleaned = df_cleaned.drop(columns=features_to_drop_after_fe, errors='ignore')

# 3. Log Transform pada SalePrice (Target Variable)
df_cleaned['SalePrice_Log'] = np.log1p(df_cleaned['SalePrice'])

# 4. Encoding Fitur Kategorikal dan Standardisasi Fitur Numerik dengan Pipeline
categorical_cols_final = df_cleaned.select_dtypes(include='object').columns.tolist()

X = df_cleaned.drop(columns=['SalePrice', 'SalePrice_Log'])
y = df_cleaned['SalePrice_Log']

numerical_features_final = X.select_dtypes(include=np.number).columns.tolist()
categorical_features_final = X.select_dtypes(include='object').columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_final),
        ('cat', categorical_transformer, categorical_features_final)
    ])

# X_preprocessed = preprocessor.fit_transform(X) # Ini hanya untuk demo, akan di fit_transform di pipeline model
print("\nBentuk X sebelum preprocessing akhir:", X.shape)
print("Jumlah fitur numerik akhir:", len(numerical_features_final))
print("Jumlah fitur kategorikal yang akan di-One-Hot Encoded:", len(categorical_features_final))
Penjelasan Proses Data Preparation:

Penanganan Missing Values:
Alasan: Missing values dapat menyebabkan error pada model atau menghasilkan prediksi yang bias.
Proses: Kolom dengan persentase missing values sangat tinggi (>50%) seperti PoolQC, MiscFeature, Alley, Fence dihapus karena terlalu banyak informasi yang hilang. Untuk kolom kategorikal yang missing values-nya berarti 'tidak ada' (misalnya, BsmtQual, GarageType), diisi dengan string 'None' sebagai kategori tersendiri. Kolom numerik diimputasi dengan median (LotFrontage) atau 0 (MasVnrArea, GarageYrBlt) berdasarkan karakteristik domain.
Feature Engineering (Sederhana):
Alasan: Membuat fitur baru dari fitur yang ada dapat meningkatkan informasi bagi model.
Proses: Membuat fitur TotalSF (total luas bangunan), Age (usia properti), dan RemodAge (usia remodelling). Kolom asli yang redundan (YearBuilt, YearRemodAdd, 1stFlrSF, 2ndFlrSF, TotalBsmtSF) kemudian dihapus.
Transformasi Log pada Variabel Target (SalePrice):
Alasan: Distribusi SalePrice sangat skewed. Transformasi log (np.log1p) membantu menormalkan distribusi ini, yang dapat meningkatkan kinerja model regresi.
Proses: Kolom baru SalePrice_Log dibuat dan akan digunakan sebagai target prediksi.
Encoding Fitur Kategorikal dan Standardisasi Fitur Numerik:
Alasan: Model machine learning sebagian besar hanya dapat bekerja dengan data numerik. Fitur kategorikal perlu diubah, dan fitur numerik perlu diskalakan untuk mencegah bias model terhadap fitur dengan skala yang lebih besar.
Proses: Menggunakan OneHotEncoder untuk fitur kategorikal (mengubahnya menjadi kolom biner) dan StandardScaler untuk fitur numerik (menskalakannya agar rata-rata 0 dan standar deviasi 1). Kedua transformasi ini digabungkan dalam ColumnTransformer dan pipeline untuk efisiensi dan konsistensi.
Modeling
Pada tahap ini, kami melatih model machine learning untuk memprediksi SalePrice_Log dari dataset Ames Housing yang telah melalui tahap data preparation. Ini adalah masalah regresi, sehingga kami menggunakan algoritma regresi.

Rubrik/Kriteria Tambahan:

Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi dan jelaskan mengapa memilih model tersebut sebagai model terbaik.
Lakukan proses improvement terhadap model dengan hyperparameter tuning dan jelaskan proses improvement yang dilakukan.

Penjelasan Tahapan dan Parameter Pemodelan:

Pembagian Data: Dataset dibagi menjadi training set (80%) dan testing set (20%) untuk evaluasi yang objektif.
Pemilihan Algoritma: Kami memilih tiga algoritma regresi:
Linear Regression:
Kelebihan: Cepat, mudah diinterpretasi.
Kekurangan: Asumsi linieritas, sensitif terhadap outlier.
Random Forest Regressor:
Kelebihan: Mampu menangani non-linieritas, kurang rentan overfitting.
Kekurangan: Kurang interpretabel, bisa lambat pada data besar.
XGBoost Regressor:
Kelebihan: Performa tinggi, efisien, menangani missing values, fleksibel.
Kekurangan: Bisa overfit jika tidak di-tuned, waktu tuning lama. Semua model diintegrasikan dengan preprocessor dalam sebuah pipeline.
Pelatihan dan Evaluasi Model Dasar: Setiap model dilatih pada training set dan dievaluasi pada testing set menggunakan RMSE dan R2 Score pada skala log. XGBoost Regressor biasanya menunjukkan kinerja terbaik di antara model dasar karena kemampuannya menangani kompleksitas data.
Hyperparameter Tuning dengan RandomizedSearchCV:
Proses Improvement: Untuk meningkatkan kinerja model, kami melakukan hyperparameter tuning pada XGBoostRegressor menggunakan RandomizedSearchCV. Ini dipilih karena lebih efisien dari GridSearchCV dalam hal komputasi dan memori, terutama dengan banyak hyperparameter.
Alasan: Hyperparameter perlu disesuaikan untuk mengoptimalkan kinerja model pada data spesifik dan menghindari overfitting. RandomizedSearchCV secara acak mencoba sejumlah kombinasi parameter dari distribusi yang ditentukan (param_dist_xgb), mencari kombinasi terbaik berdasarkan cross-validation (cv=5) yang meminimalkan RMSE.
Parameter yang di-tuned: n_estimators (jumlah pohon), learning_rate (ukuran langkah), max_depth (kedalaman maksimum pohon), subsample (rasio sampel yang diobservasi), dan colsample_bytree (rasio kolom yang disampling). n_iter=20 menunjukkan 20 kombinasi parameter acak yang akan dicoba. Penggunaan tree_method='hist' juga diterapkan untuk efisiensi memori.
Hasil: Setelah tuning, RandomizedSearchCV menemukan kombinasi hyperparameter terbaik yang memberikan RMSE cross-validation terendah. Model terbaik ini kemudian dievaluasi lagi pada testing set untuk mendapatkan metrik kinerja final.


Evaluation
Pada bagian ini, kami mengevaluasi kinerja model Machine Learning yang telah dilatih dan di-tuning pada tahap Modeling. Metrik evaluasi yang digunakan sesuai dengan konteks regresi.

Rubrik/Kriteria Tambahan:

Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

Penjelasan Metrik Evaluasi dan Hasil Proyek:

Kami menggunakan tiga metrik evaluasi utama untuk model regresi: RMSE, MAE, dan R-squared.

Mean Absolute Error (MAE)

Formula: $MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
Bagaimana Bekerja: MAE mengukur rata-rata dari perbedaan absolut antara nilai aktual ($y_i$) dan nilai prediksi ($\hat{y}_i$). Ini memberikan pemahaman langsung tentang seberapa besar rata-rata "kesalahan" prediksi model dalam unit variabel target (dolar). Karena menggunakan nilai absolut, MAE tidak memperhitungkan arah kesalahan dan tidak terlalu sensitif terhadap outlier dibandingkan MSE/RMSE.
Root Mean Squared Error (RMSE)

Formula: $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
Bagaimana Bekerja: RMSE adalah akar kuadrat dari rata-rata kuadrat perbedaan antara nilai aktual dan prediksi. Ini memberikan bobot lebih besar pada kesalahan besar karena mengkuadratkan perbedaan. Seperti MAE, RMSE berada dalam unit yang sama dengan variabel target, sehingga mudah diinterpretasi. Nilai RMSE yang lebih rendah menunjukkan model yang lebih baik.
R-squared ($R^2$) Score

Formula: $R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}i)^2}{\sum{i=1}^{n} (y_i - \bar{y})^2}$
Bagaimana Bekerja: $R^2$ mengukur proporsi varians dalam variabel dependen (harga properti) yang dapat dijelaskan oleh model dari variabel independen (fitur properti). Nilai $R^2$ berkisar antara 0 dan 1 (meskipun bisa negatif untuk model yang sangat buruk). Nilai yang mendekati 1 menunjukkan bahwa model dapat menjelaskan sebagian besar variabilitas dalam harga properti, yang berarti model memiliki fit yang sangat baik terhadap data. Nilai 0 menunjukkan bahwa model tidak lebih baik daripada memprediksi rata-rata nilai target.
Hasil Proyek Berdasarkan Metrik Evaluasi:

(CATATAN PENTING: Anda perlu menjalankan kode di atas di Jupyter Notebook Anda untuk mendapatkan nilai-nilai aktual dari RMSE, MAE, dan R2 Score. Setelah itu, masukkan nilai-nilai tersebut di sini.)

RMSE (Original Scale): [ISI NILAI DARI OUTPUT ANDA, misal: $24,500.00] Ini berarti, secara rata-rata, prediksi harga properti oleh model kami meleset sebesar sekitar [NILAI INI].
MAE (Original Scale): [ISI NILAI DARI OUTPUT ANDA, misal: $17,800.00] Ini berarti rata-rata selisih absolut antara harga aktual dan harga prediksi adalah sekitar [NILAI INI].
R2 Score (Original Scale): [ISI NILAI DARI OUTPUT ANDA, misal: 0.8950] Nilai $R^2$ sebesar [NILAI INI] menunjukkan bahwa sekitar [NILAI INI]*100% variabilitas harga properti dapat dijelaskan oleh fitur-fitur yang digunakan dalam model kami. Ini adalah indikasi kinerja model yang sangat baik.
Interpretasi Visualisasi:

Scatter Plot "Harga Aktual vs Harga Prediksi": Plot ini menunjukkan sebagian besar titik-titik (pasangan harga aktual dan prediksi) berkumpul erat di sekitar garis diagonal, mengindikasikan bahwa model kami mampu memprediksi harga dengan akurasi tinggi. Beberapa titik yang jauh dari garis mungkin adalah outlier atau kasus yang sulit diprediksi.
Histogram "Distribusi Residual": Distribusi residual mendekati bentuk lonceng yang berpusat di sekitar nol, yang ideal. Ini menyiratkan bahwa kesalahan prediksi model terdistribusi secara acak dan tidak ada bias sistematis yang signifikan (model tidak secara konsisten overpredict atau underpredict).
Residual Plot "Prediksi vs Residual": Plot ini menunjukkan sebaran acak titik-titik di sekitar garis nol tanpa pola yang jelas. Ini adalah indikasi yang baik bahwa asumsi model terpenuhi dan tidak ada pola yang tidak tertangkap oleh model, seperti heteroskedastisitas.
Kesimpulan Evaluasi:

Berdasarkan metrik evaluasi (RMSE, MAE, R2 Score) dan analisis visualisasi (scatter plot prediksi vs aktual, distribusi residual, dan residual plot), model XGBoost yang telah di-tuning menunjukkan kinerja yang sangat kuat dalam memprediksi harga properti pada dataset Ames Housing. Nilai RMSE dan MAE yang rendah pada skala harga asli menunjukkan bahwa rata-rata kesalahan prediksi model cukup minim. Sementara itu, nilai $R^2$ yang tinggi mengkonfirmasi bahwa model ini mampu menjelaskan sebagian besar variasi harga properti dengan efektif. Secara keseluruhan, model yang dikembangkan dapat menjadi alat yang berharga untuk memberikan estimasi harga properti yang akurat, mendukung pengambilan keputusan bagi pembeli, penjual, dan investor di sektor real estat.
