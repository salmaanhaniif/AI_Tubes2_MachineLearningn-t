<br/>
<h1 align="center"> Implementasi Algoritma Pembelajaran Mesin Decision Tree Learning, Logistic Regression, dan Support Vector Machine pada Dataset UNSW-NB15</h1>

<br/>

> Tugas Besar 2 IF3170 - Inteligensi Artifisial
> By Kelompok Machine Learningn't - K02 - IF'23

<br/>

## Deskripsi Program

Repository ini berisikan implementasi beberapa algoritma pembelajaran mesin, yaitu DTL, logistic regression, dan SVM, from scratch menggunakan bahasa Python yang kemudian digunakan untuk memprediksi apakah sebuah mahasiswa `Dropout`, `Graduate`, atau `Enrolled` berdasarkan berbagai atribut/label yang diberikan pada dataset UNSW-NB15.

<br/>

## Requirements
- Python ≥ 3.8
- Berbagai library python yang terdiri dari
  - numpy
  - pandas
  - scikit-learn
  - numba
  - matplotlib
<br/>

## Cara Instalasi dan Penggunaan
### Instalasi

1. Clone repository

``` bash   
git clone https://github.com/salmaanhaniif/AI_Tubes2_MachineLearningn-t.git
```

2. Install dependencies (jika belum terinstall)

``` bash   
pip install numpy pandas scikit-learn numba matplotlib
```

Script utama berada di folder `src/`

### Penggunaan

#### Cara Penggunaan Algoritma DTL
``` python
from DTL import DecisionTreeLearning
import pandas as pd

df = pd.read_csv('dataset.csv')
X = [isi dengan seluruh value X yang digunakan]
y = [isi dengan seluruh value target y]

model = DecisionTreeLearning(max_depth=10)
model.fit(X, y)

pred = model.predict(X)
model.save_model("dtl_model.pkl")
model = DecisionTreeLearning().load_model("dtl_model.pkl")
```

#### Cara Penggunaan Algoritma Logistic Regression
``` python
from LogisticRegression import LogisticRegression
import pandas as pd

df = pd.read_csv('dataset.csv')
X = [isi dengan seluruh value X yang digunakan]
y = [isi dengan seluruh value target y]

model = LogisticRegression(learning_rate=0.01, n_iterations=200, mini_batch=True)
model.fit(X, y)

pred = model.predict(X)
model.save_model("logreg.pkl")
model = LogisticRegression.load_model("logreg.pkl")

# untuk generate gif training
model.generate_training_gif(X, y, output_path='training.gif')
```

#### Cara Penggunaan Algoritma SVM
``` python
from SVM import MulticlassSVM
import pandas as pd

df = pd.read_csv('dataset.csv')
X = [isi dengan seluruh value X yang digunakan]
y = [isi dengan seluruh value target y]

model = MulticlassSVM(C=1.0, kernel='linear')
model.fit(X, y)

pred = model.predict(X)
model.save_model("svm.pkl")
model = MulticlassSVM.load_model("svm.pkl")
```

### Pembagian Tugas

| NIM | Nama | Tugas |
| :---: | :---: | :---: |
| 13523050 | Mayla Yaffa Ludmilla | Implementasi DTL + Evaluation |
| 13523056 | Salman Hanif | Implementasi DTL + Preprocessing |
| 13523058 | Noumisyifa Nabila Nareswari | Implementasi Logistic Regression + Evaluation |
| 13523080 | Diyah Susan Nugrahani | Implementasi SVM + Logistic Regression |
| 13523096 | Muhammad Edo Raduputu Aprima | Implementasi SVM + Preprocessing |
