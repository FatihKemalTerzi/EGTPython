import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return  data
df = load_application_train()
df.head()

def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()

###Aykırı değerleri hesaplama

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1

up = q3 + iqr * 1.5 ##64.8125
low = q1 - 1.5 * iqr##-6.6875

df[(df["Age"] < low) | (df["Age"] > up)] #age değişkenindeki aykırı değerleri bulduk

#Aykırı değer varmı yokmu? any methodu ile içerisinde aykırı değerin hiç olup olmadığını sorgulayıp True veya False döndürebiliriz.
df[~(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

##işlemleri fonksiyonlaitırmak

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquartilerange = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquartilerange
    low_limit = quartile1 - 1.5* interquartilerange
    return  low_limit,up_limit

for col in df.columns:
    if df[col].dtypes != "O":
        lower, upper = outlier_thresholds(df, col)
        print(col, upper , lower)

#Aykırı değer var mı yok mu diye bakmak:

def check_outlier(dataframe, col):
    low, up = outlier_thresholds(dataframe, col)
    if dataframe[(dataframe[col] < low) | (dataframe[col] > up)].any(axis=None) and dataframe[col].dtypes != "O":
        return True
    else:
        return False
check_outlier(df,"Age")

def grab_col_names(df,cat_th=10,car_th=20):
    cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

    cat_but_car = [col for col in cat_cols if df[col].nunique() > car_th]

    num_cols = [col for col in df.columns if df[col].dtypes != "O"]

    num_but_cat = [col for col in num_cols if df[col].nunique() < cat_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, cat_but_car,num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)

## Çok değişkenli aykırı değer analizi
#tek başına olunca aykırı olamayacak bazı değerler beraber alınınca aykırı olabilir buna çok değişkenli aykırı değerler denir

df = sns.load_dataset("diamonds")
df = df.select_dtypes(include=['float', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df,col))

low, up = outlier_thresholds(df, "carat")

#Eksik Değerleri Yakalama

df = load()
df.head()

#veri setinde herhangi bir eksik değer varsa
df.isnull().any()

#Değişkenlerdeki eksik değer sayısı ⇒
df.isnull().sum()

#Değişkenlerdeki tam değer sayısı⇒
df.notnull().sum()

#Veri setindeki toplam eksik değer sayısı⇒
df.isnull().sum().sum()

#En az bir tane eksik değere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

#Tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

#Değişkenlerin eksik değerlerinin % olarak oranına ulaşmak istiyorsak
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

#Eksik değerleri seçiyoruz
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

#Eksik değerleri seçme işlemini fonksiyonlaştırıyoruz
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    n_miss= dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1,keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


#Eksik Değer Problemini Çözme
#Çözüm 1: Hızlıca silmek
#Eksik değer olan satırları dropna() ile silebiliriz
df.dropna()

#Çözüm 2: Basit atama yöntemleri ile doldurmak
#Burada ilgili değişkenin ortalama veya medyan değerleri ile doldurduk
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())
#Veya sabit bir değer ile doldurabiliriz burada sıfır değeri ile doldurduk
df["Age"].fillna(0)(df["Age"].median())

#İşlemi Fonksiyonlaştırma
#Eğer değişken sayısal bir değişkense o değişkenin ortalama değerleri ile eksik değerleri doldur
df.apply(lambda x:x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

#Kategorik değişkenler için doldurma işlemi
#Kategorik değişkenler için en mantıklı dıldurma yöntemi modunu almaktir yani o değişkenin modu ile doldurmaktır

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

#Eğer bu değişkenin içerisindeki ifadeyi string bir ifade ile (mesela 'missing' ifadesi) doldurmak istersek
df["Embarked"].fillna("missing")

#Bu işlemi otomatikleştirirsek
#Eğer ilgili değişken tipi kategorik ise ve eşsiz değişken sayısı 10 dan küçük eşitse eksik değerleri o değişkenin modu ile doldur
df.apply(lambda x:x.fillna(x.mode()[0]) if (x.dtype == "0" and len(x.unique()) <= 10) else x, axis=0)


