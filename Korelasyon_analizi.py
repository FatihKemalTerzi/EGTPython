# 5. Korelasyon Analizi (Analysis of Correlation)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 1:-1]
print(df.head())
df.shape
#sadece ihtiyaç olduğunda kullanılır analiz aracı olarak

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]

#korelasyonları hesaplamak çin corr fonksiyonu kullanılır
corr = df[num_cols].corr()

#yüksek Korelasyonlu Değişkenlerin Silinmesi

cor_matrix = df.corr().abs()#burada korelasyonun artı veya eksi olması benim için farketmediğinden mutlak değerini aldım

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
#bu sütunlarda herhangi birisi %90 dan yani 0.90 dan büyük ise onları silmesi için drop_list içine attık
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]

df.drop(drop_list, axis=1)#silme işlemi


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    #önce korelasyon oluşutuyoruz
    cor_matrix = corr.abs()
    #mutlak değerini alıyoruz
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    #köşegen elemanlarına göre düzeltme işlemi yaptık
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    #belirli bir korelasyonun üzerinde olanları silmek için bir liste oluşturduk
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


high_correlated_cols(df)