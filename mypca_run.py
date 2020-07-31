import pandas as pd
import numpy as np
import mypca

df = pd.read_csv("test_df_4dim.csv",header=None)

data = df.drop(df.columns[0],axis=1)
data = np.array(data)
data = data.astype(np.float64)

#===param===
# n_components (if None => Not compression)
# normalizing  defalt is True
pca = mypca.pca(n_components=None)
v = pca.fit(data)

#固有値
print(pca.eigen_value)
#固有ベクトル
print(pca.eigen_vector)
#分散共分散行列
print(pca.cov_ver_matrix)

#主成分得点（ソート済み）
print(pca.p_component)
#累積寄与率
print(pca.ex_va)


