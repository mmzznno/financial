#ライブラリ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz
from sklearn.svm import SVC
from sklearn import metrics

# データ読み込み
df_train = pd.read_csv('../input/train.csv',index_col='id')
df_test = pd.read_csv('../input/test.csv',index_col='id')

#print(df_train.shape) 
#train[20346rows x 15 columns]
#test[6782rows x 14columns]

#投稿用ファイル
submit = pd.read_csv("../input/sample_submission.csv", header=None)

#print(submit) 
#[6782rows x 2]

#学習用と評価用が分別できるようフラグを立てる
df_train["flag"] = True
df_test["flag"] = False

#一旦、学習用データと評価用データを結合
df = pd.concat([df_train, df_test], axis=0, sort =True)

#print(df.shape)
#(27128, 16)

#print(df.head(30))

#仮説1
#前回申し込みによるクロス集計
cross = pd.crosstab(df['poutcome'], df['y'], margins=True)
print(cross)

#前回仕事クロス集計
cross = pd.crosstab(df['job'], df['y'], margins=True)
print(cross)

# データのダミー変数化
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

# 説明変数をdata_Xに、目的変数をdata_yに代入
data_X, data_y = df_train.drop('y', axis=1), df_train['y']

# train_test_splitのインポート
from sklearn.model_selection import train_test_split

# 学習データと評価データにデータを分割
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, test_size=0.30, random_state=0)

# 決定木モデルのインポート
from sklearn.tree import DecisionTreeClassifier as DT

# 決定木モデルの準備
tree = DT(max_depth = 3, random_state = 0)

# 決定木モデルの学習
tree.fit(train_X, train_y)

# 重要度の表示
print(tree.feature_importances_)

# 重要度に名前を付けて表示
print( pd.Series(tree.feature_importances_, index=train_X.columns) )

## 重要度の表示
print(tree.feature_importances_)
#poutcome_success       0.347722
#duration               0.652278

# 評価用データの予測
pred_y1 = tree.predict_proba(test_X)[:,1]

# 予測結果の表示
#print(pred_y1)

# AUCの計算
auc1 = roc_auc_score(test_y, pred_y1)

# 偽陽性率、真陽性率、閾値の計算
fpr, tpr, thresholds = roc_curve(test_y, pred_y1)

# ラベル名の作成
roc_label = 'ROC(AUC={:.2}, max_depth=3)'.format(auc1)
# ROC曲線の作成
plt.plot(fpr, tpr, label=roc_label)
# 対角線の作成
plt.plot([0, 1], [0, 1], color='black', linestyle='dashed')
# グラフにタイトルを追加
plt.title("ROC")
# グラフのx軸に名前を追加
plt.xlabel('FPR')
# グラフのy軸に名前を追加
plt.ylabel('TPR')
# x軸の表示範囲の指定
plt.xlim(0, 1)
# y軸の表示範囲の指定
plt.ylim(0, 1)
# 凡例の表示
plt.legend()
# グラフを表示
plt.show()

# 決定木描画ライブラリのインポート
from sklearn.tree import export_graphviz

# 決定木グラフの出力
export_graphviz(tree, out_file="tree.dot", feature_names=train_X.columns, class_names=["0","1"], filled=True, rounded=True)

# 決定木グラフの表示
from matplotlib import pyplot as plt
from PIL import Image
import pydotplus
import io

g = pydotplus.graph_from_dot_file(path="tree.dot")
gg = g.create_png()
img = io.BytesIO(gg)
img2 = Image.open(img)
plt.figure(figsize=(img2.width/100, img2.height/100), dpi=100)
plt.imshow(img2)
plt.axis("off")
plt.show()

# グリッドサーチのインポート
from sklearn.model_selection import GridSearchCV

# 決定木モデルの準備
Tree = DT(random_state=0)

# パラメータの準備
parameters= {'max_depth':[2, 3, 4, 5, 6, 7, 8, 9, 10]}

# グリッドサーチの設定
gcv = GridSearchCV(Tree, parameters, cv=5, scoring='roc_auc', return_train_score=True)

# グリッドサーチの実行
gcv.fit(train_X, train_y)

train_score = gcv.cv_results_['mean_train_score']
test_score = gcv.cv_results_['mean_test_score']
print(train_score)
print(test_score)

#提出データの予測
#X_test2 = df_test[df["flag"] == False]
#X_test2 = X_test2.drop(["flag"], axis = 1)

# 最適なパラメータの表示
print( gcv.best_params_ )

# 最適なパラメータで学習したモデルの取得
best_model = gcv.best_estimator_

# 評価用データの予測
pred_y3 = best_model.predict_proba(df_test)[:,1]

submit[1] = pred_y3
submit.to_csv("submit.csv", index=False, header=False)

print(submit)
