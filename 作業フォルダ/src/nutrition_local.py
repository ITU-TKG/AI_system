#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install nbconvert')


# In[534]:


get_ipython().system('jupyter nbconvert --to python nutrition_local.ipynb')


# In[535]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


get_ipython().system('pip install -r requirements.txt')


# In[ ]:


import pandas as pd

path = "db_nutrition.csv"

df = pd.read_csv(path)


# #前処理

# In[550]:


df = df.fillna(0)


# In[551]:


df = df.rename(columns={
    'Unnamed: 0': '食品群',
    'Unnamed: 1': '食品番号',
    'Unnamed: 2': '索引番号',
    '単位': "食品名",
    '%': "廃棄率(%)",
    'kJ': "エネルギー(kJ)",
    'kcal': "エネルギー(kcal)",
    '(…………… g ………………)': "水分(g)",
    'Unnamed: 8': 'アミノ酸組成によるたんぱく質(g)',
    'Unnamed: 9': 'たんぱく質(g)',
    'Unnamed: 10': '脂肪酸のトリアシルグリセロール当量(g)',
    'mg': 'コレステロール(mg)',
    '(……………………………… g ……………………………)': '脂質(g)',
    'Unnamed: 13': '利用可能炭水化物(単糖当量)(g)',
    'Unnamed: 14': 'Name1',
    'Unnamed: 15': '利用可能炭水化物(質量計)(g)',
    'Unnamed: 16': '差引き法による利用可能炭水化物(g)',
    'Unnamed: 17': 'Name2',
    'Unnamed: 18': '食物繊維総量(g)',
    'Unnamed: 19': '糖アルコール(g)',
    'Unnamed: 20': '炭水化物(g)',
    'Unnamed: 21': '有機酸(g)',
    'Unnamed: 22': '灰分(g)',
    '(…………………………… mg ……………………………)': 'ナトリウム(mg)',
    'Unnamed: 24': 'カリウム(mg)',
    'Unnamed: 25': 'カルシウム(mg)',
    'Unnamed: 26': 'マグネシウム(mg)',
    'Unnamed: 27': 'リン(mg)',
    'Unnamed: 28': '鉄(mg)',
    'Unnamed: 29': '亜鉛(mg)',
    'Unnamed: 30': '銅(mg)',
    'Unnamed: 31': 'マンガン(mg)',
    'Unnamed: 32': 'Name3',
    '(…………………………………… μg………………………………………)': 'ヨウ素(μg)',
    'Unnamed: 34': 'セレン(μg)',
    'Unnamed: 35': 'クロム(μg)',
    'Unnamed: 36': 'モリブデン(μg)',
    'Unnamed: 37': 'レチノール(μg)',
    'Unnamed: 38': 'a-カロテン(μg)',
    'Unnamed: 39': 'β-カロテン(μg)',
    'Unnamed: 40': 'β-クリプトキサンチン(μg)',
    'Unnamed: 41': 'β-カロテン当量(μg)',
    'Unnamed: 42': 'レチノール活性当量(μg)',
    'Unnamed: 43': 'ビタミンD(μg)',
    '(………… mg …………)': 'a-トコフェロール(mg)', 
    'Unnamed: 45': 'β-トコフェロール(mg)',
    'Unnamed: 46': 'γ-トコフェロール(mg)',
    'Unnamed: 47': 'σ-トコフェロール(mg)',
    'μg': 'ビタミンK(μg)',
    '(…………… mg ……………)': 'ビタミンB1(mg)',
    'Unnamed: 50': 'ビタミンB2(mg)',
    'Unnamed: 51': 'ナイアシン(mg)',
    'Unnamed: 52': 'ナイアシン当量(mg)',
    'Unnamed: 53': 'ビタミンB6(mg)',
    '(…… μg……)': 'ビタミンB12(μg)',
    'Unnamed: 55': '葉酸(μg)',
    'mg.1': 'バントテン酸(mg)',
    'μg.1': 'ビオチン(μg)',
    'mg.2': 'ビタミンC(mg)',
    '(……g……)': 'アルコール(g)',
    'Unnamed: 60': '食塩相当量(g)',
    'Unnamed: 61': 'Name4',
    'Unnamed: 62': 'Name5',
    'Unnamed: 63': 'Name6',
    'Unnamed: 64': 'Name7'
})


# In[552]:


# 一行目（インデックス0）の削除
df = df.drop(index=0).reset_index(drop=True)


# In[553]:


#Name削除
df = df.drop(columns=[col for col in df.columns if 'Name' in col])


# In[554]:


# データフレーム内の全てのセルの値で () と - を 置換
df = df.applymap(lambda x: x.replace('(', '').replace(')', '').replace('-', '0').replace('Tr', '0') if isinstance(x, str) else x)


# In[555]:


# 空白文字（スペースやタブなど）を含むセルの空白文字を削除
import re

df = df.applymap(lambda x: re.sub(r'\s+', '', x) if isinstance(x, str) else x)


# In[556]:


df["食塩相当量(g)"].head(10)


# In[557]:


# すべての列（食品名以外）をfloat型に変換（object型のみ対象）
for col in df.columns:
    if col != '食品名':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 変換後の型を確認
print(df.dtypes)


# In[558]:


df.to_csv("new_nutrition_db.csv", index=False)


# #相関分析

# ad

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_1 = df_ad.drop(columns=['Food_Item','Category','Meal_Type','Date','User_ID'])
# 相関行列を計算
correlation_matrix = df_1.corr()
print(correlation_matrix)

# ヒートマップで可視化
plt.figure(figsize=(12, 10))  # 図のサイズを指定
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()


# db

# In[559]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 日本語フォントを明示的に指定
# Windowsの場合の例（MS Gothic）
plt.rcParams['font.family'] = 'MS Gothic'

# 相関行列を計算
df_numeric = df.select_dtypes(exclude=['object'])

correlation_matrix = df_numeric.corr()
print(correlation_matrix)

# ヒートマップで可視化
plt.figure(figsize=(12, 10))  # 図のサイズを指定
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix', fontname='MS Gothic')
plt.show()


# In[ ]:


get_ipython().system('pip install transformers')


# In[522]:


get_ipython().system('pip freeze > requirements.txt')


# 健康度 計算

# In[425]:


#不健康ラベルに使う要素の相関分析
corr = df[["ナトリウム(mg)","脂質(g)","コレステロール(mg)"]].corr(method='pearson')
corr


# 可食部100g当たりの食品のデータ

# #一日
# コレステロール750mg,
# ナトリウム4.5g/1000kcal,
# 脂質25%/ene

# #人間は一日2kgの食事をとると仮定

# In[594]:


#式定義

df["コレステロール(mg)"] / 100 > 750 / 2000
df["ナトリウム(mg)"]*1000 / df["エネルギー(kcal)"] > 4.5 / 1000
df["脂質(g)"] / df["エネルギー(kcal)"] > 0.25


# In[595]:


import pandas as pd

def add_unhealthy_label(df: pd.DataFrame) -> pd.DataFrame:

    cond1 = df["コレステロール(mg)"] / 100 > 750 / 2000
    cond2 = df["ナトリウム(mg)"]*1000 / df["エネルギー(kcal)"] > 4.5 / 1000
    cond3 = df["脂質(g)"] / df["エネルギー(kcal)"] > 0.25

    # Trueの合計が2つ以上なら不健康
    conditions_sum = cond1.astype(int) + cond2.astype(int) + cond3.astype(int)
    df["unhealthy"] = (conditions_sum >= 2).astype(int)

    return df


# In[596]:


df = add_unhealthy_label(df)
print(df[["unhealthy"]].value_counts())

df.to_csv("feature&label_nutrition_db.csv", index=False)


# In[597]:


from sklearn.model_selection import train_test_split
from transformers.trainer_utils import set_seed

feature_columns = [col for col in df.columns if col not in ['食品名', '食品群', '食品番号', '索引番号', '廃棄率(%)', 'コレステロール(mg)', '食塩相当量(g)', 'ナトリウム(mg)', '脂質(g)', 'エネルギー(kcal)', 'unhealthy']]
data = df[feature_columns].values
label = df['unhealthy']


set_seed(42)


# 分類

# In[598]:


x_train,x_val,y_train,y_val = train_test_split(data, label, test_size=0.3, random_state=42)


# In[599]:


# 欠損値を0で埋める
x_train = pd.DataFrame(x_train).fillna(0).values
x_val = pd.DataFrame(x_val).fillna(0).values


# In[600]:


print(y_train[0:10])
print(y_train.shape)
print(df.shape)


# In[601]:


# クラス分布の確認
print(label.value_counts())
print("クラス数:", label.nunique())


# In[602]:


x_train


# In[ ]:


from sklearn.model_selection import GridSearchCV

search_gs = {
    "max_depth": [None, 5, 25],
    "n_estimators": [150, 180],
    "min_samples_split": [4, 8, 12],
    "max_leaf_nodes": [None, 10, 30],
}

model_gs = RandomForestClassifier(random_state=42)

gs = GridSearchCV(
    model_gs,
    search_gs,
    cv=5,
    iid=False
)

gs.fit(x_train, y_train)
print("Best parameters:", gs.best_params_)
print("Best score:", gs.best_score_)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

model = RandomForestClassifier(
                               class_weight='balanced',
                               criterion='gini',  
                               n_estimators=100,
                               random_state=42,
                               max_depth=10,
                               max_features='sqrt',
                            )
model.fit(x_train, y_train)
y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'RandomForest Accuracy:{accuracy:.3f}')


# In[ ]:


get_ipython().system('pip install pydotplus graphviz')


# In[604]:


from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz

estimators = model.estimators_
dot_data = export_graphviz(estimators[0],
                           out_file=None,
                           filled=True,
                           rounded=True,
                           class_names=['Healthy', 'Unhealthy'],
                           feature_names=feature_columns,
                           special_characters=True,
                           fontname='MS Gothic'
                        )

graph = graph_from_dot_data(dot_data)
graph.progs = {'dot': 'C:\\Users\\bi23014\\windows_10_cmake_Release_Graphviz-12.2.1-win64\\Graphviz-12.2.1-win64\\bin\\dot.exe'}  # Graphvizのパスを指定
graph.write_png('decision_tree.png')


# ＃評価

# In[605]:


#score - 汎用性能・テスト性能を測る
y_train_score = model.score(x_train,y_train)
print("Train Score: ", y_train_score)

y_val_score = model.score(x_val,y_val)
print("Validation Score: ", y_val_score)


# In[606]:


from sklearn.metrics import classification_report
print(classification_report(y_val, y_pred, digits=3))


# In[607]:


#ROC曲線とAUCの計算
from sklearn.metrics import roc_curve, auc

y_score = model.predict_proba(x_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, y_score)
roc_auc = auc(fpr, tpr)
print(roc_auc)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()



# In[608]:


# 混同行列を計算して表示
cm = confusion_matrix(y_val, y_pred)
# 混同行列をヒートマップとして表示
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Healthy', 'Unhealthy'],
    yticklabels=['Healthy', 'Unhealthy'],
    annot_kws={"size": 10}
    )

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.show()

