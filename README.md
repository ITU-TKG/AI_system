# AI_system
講義　知的情報システム開発2　の作業フォルダです

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

