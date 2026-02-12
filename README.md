# AI_system
講義　知的情報システム開発2　の作業フォルダです  
この講義は2025/4 ~ 2025/7 までの内容です  

## フォルダ説明
作業フォルダ内のフォルダ説明  
submit --- 最終提出物をまとめた親フォルダです  
以下submit内  
data --- 使用したrawデータと加工後データに加え、不健康ラベルが付与された食材を羅列したファイルと参考元urlを格納したフォルダ  
result --- cm分析結果、rf及びgbmモデルそれぞれの決定木の予測結果、roc_curveの結果を格納したフォルダ

### srcフォルダ内のファイル説明
提出フォルダ内には説明書の他に8個のファイルが存在する。それぞれの内容をこのセクションで要約する。  
・all_fi_gbm.ipynb  
--- 全特徴量を学習に回したlightGBMのソースコード  
・all_fi_rf.ipynb  
--- 全特徴量を学習に回したrandomforestのソースコード  
・del_top3_fi_rf.ipynb  
--- 特徴量重要度が大きかった上位3つの特徴量を学習データから削除して学習を行う、randomforestのソースコード  
・del_top5_fi_rf.ipynb  
--- 特徴量重要度が大きかった上位5つの特徴量を学習データから削除して学習を行う、randomforestのソースコード  
・del_top_fi_gbm.ipynb  
--- 特徴量重要度が一番大きかった特徴量を学習データから削除して学習を行う、lightGBMのソースコード  
・nutrition_local_20features.ipynb  
--- 特徴量重要度が大きかった上位20個の特徴量以外を学習データから削除して学習を行う、randomforestのソースコード  

## 構想についてなど
### 目的について
本講義は何かしら自主的に問題発見をしてそれに対して機械学習を用いて成果物を作ることが目的  
私は栄養バランスの良い献立を考えることを目標に機械学習を用いて健康不健康を判断する分類器作成を目指した。  

データ等の観点から健康不健康の分類が困難であったため「食べようとしている食事が不健康かどうか判断したい」という理念のもと”材料単位”で栄養素データを取得し学習させる方法を用いた。  

### 事前準備
学習にあたって不健康ラベルをルールベースによって付与することにした  
方法：  

一日摂取上限により不健康ラベルを付与する　　
摂取上限 --- コレステロール750mg, ナトリウム4.5g/1000kcal, 糖類/エネルギーkcal = 10.7%

留意事項  
・不健康ラベルは0,1でつけており、不健康は1とする
・一日単位での栄養摂取基準を適用  
・可食部100g当たりの食品データを用いる  
・一日当たり2kgの食事をとると仮定  
・20代の男性で摂取上限を設定  


次にデータの前処理について  

データInfo  
・文部科学省 日本食品標準成分表  
・総数2500件程度  

処理  
・列名変更  
・Nan値処理  
・特殊文字変換...etc  

↓↓↓ 以下図はラベルの結果 ↓↓↓   


<img width="1999" height="1016" alt="image" src="https://github.com/user-attachments/assets/2e8fe215-2e09-4ab2-9af4-431c14d3a744" />  


ナトリウム過剰が多い  
閾値を超える度合い（画像左3つの図）を見るとコレステロール、炭水化物に対してナトリウムの振れ幅が大きかった  
塩分過多を特に反映したラベルとなったといえる  

### 学習  
使用モデル  
・Randomforest   
・lightGBM  
どちらも決定木を用いた機械学習の分類器  
モデルの違い  
randomforest:  
バギングによって複数の決定木から予測結果を平均化または多数決で結合する(並列処理)過学習に強いモデル  

lightgbm:  
ブースティングにより複数の弱い決定木を構築して前のモデルの間違えた部分に重みを置いて次モデルを改善する(直列処理)バイアスを減らすことができるモデル  

lightgbmは精度向上に向いているが計算負荷が高い、randomforestは安定性が高く比較的簡単であるが精度は劣る  

それぞれのモデルを使用して特徴量重要度によって特徴量を変更することによりモデルの性能を比較した  

## 学習結果 分類器の性能比較  
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/bdfd5f66-4bb2-45b9-9c09-29f6e6e5e3ff" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ddd494b1-412c-45ba-9ac1-7847570685f2" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/c6ae80d7-df65-41fc-9ef6-728e79c7ed84" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/d6073acf-444e-466a-8671-1c61ce36aed1" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/59254484-ea81-472c-ba0b-9389beec2285" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ecc80928-3ea8-4e39-a944-25d08b931a50" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/939a4f02-cb65-4142-8cb5-534d571ec7a0" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ff855520-c4a6-4e26-b28b-6023d5e5f52b" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/9c23980e-aed4-433d-beb2-bd2e5ba15ca7" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/e3c33861-8f89-4ecd-8004-05a623c2fa53" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/7dd524ab-49d4-4a0c-a39e-f3eb0966db95" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/40773e80-adaa-4f01-be74-f603d40e67d4" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/ba3f441f-9c4d-4a64-aa44-27ddb733190f" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/d6d79613-207b-4618-9dc3-ebc03100ce6d" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/41751d46-097e-4765-9ce1-7ed3550bf798" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/04ced8bf-b750-44f9-a4e2-19045ad8e9a8" />  

<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/da4971f8-e335-499d-80ce-2be27536c3c1" />  

















