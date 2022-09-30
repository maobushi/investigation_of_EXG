import numpy as np
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器
from pandas import plotting 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



df = pd.read_csv(filepath)

print(df.head())


timeStamp = [0,
             43,
             109,
             290,
             412,
             487,
             603,
             688,
             771,
             878,
             975,
             1058,
             1160]


convertTS = [n*500 for n in timeStamp]

print("CTS is :",convertTS)
'''
for i in range(10):
  df_tmp[i] = df[timeStamp[i]:timeStamp[i+1]]
'''

fig = make_subplots(rows=1, cols=1)


def print_graph(title,titlename):
 fig.add_trace(go.Scatter(x=np.arange(convertTS[0],convertTS[1]),
                         y=df[title],
                         mode='lines',
                         name=titlename,
                        ), 
              row=1, col=1)


print_graph(' EOG','EOG')
print_graph(' EOG.1','EOG.1') 
print_graph(' 6-REF','T8') 
print_graph(' EMG','EMG') 
print_graph(' EMG.1','EMG.1') 
print_graph(' EMG.2','EMG.2') 


fig.show()

X = np.vstack([df[' EOG'],df[' 6-REF']]).T


corr = np.corrcoef(df.values.T)

# ヒートマップとして可視化
hm   = sns.heatmap(
                 corr,                         # データ
                 annot=True,                   # セルに値入力
                 fmt='.2f',                    # 出力フォーマット
                 annot_kws={'size': 8},        # セル入力値のサイズ
                 yticklabels=list(df.columns), # 列名を出力
                 xticklabels=list(df.columns)) # x軸を出力

plt.tight_layout()
plt.show()

'''
sns.pairplot(df)
plt.tight_layout()
plt.show()
'''


Y = np.array(df[' 6-REF'])
 
# 説明変数(X)
col_name = [' EMG',' EMG.1',' EMG.2',' EOG',' EOG.1',' 6-REF']
X = np.array(df[col_name])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# モデル構築　
model = LinearRegression()
 
# 学習
model.fit(X_train, Y_train)

# 回帰係数
coef = pd.DataFrame({"col_name":np.array(col_name),"coefficient":model.coef_}).sort_values(by='coefficient')
 
# 結果
print("【回帰係数】", coef)
print("【切片】:", model.intercept_)
print("【決定係数(訓練)】:", model.score(X_train, Y_train))
print("【決定係数(テスト)】:", model.score(X_test, Y_test))
