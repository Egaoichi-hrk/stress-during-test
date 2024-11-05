import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import statsmodels.api as sm
import matplotlib.font_manager as fm

# 日本語フォントの設定
plt.rcParams['font.family'] = 'IPAexGothic'  # 'IPAexGothic'はインストール済みの日本語フォント
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号が文字化けしないように設定

st.header('試験中に感じるストレスとその成績結果の関係性')

st.subheader('単回帰分析を行う')
df = pd.read_csv('data.csv')
st.write('生徒数10名、HRの値は一つの試験中に観測された一分間ごとの心拍数の平均、中間1と中間2の成績は100点満点、最終試験は200点満点')
st.write(df)

st.subheader('中間1結果')
x = df['中間１.HR']
y = df['中間1成績']
correlation = np.corrcoef(x, y)[0, 1]
st.write(f'相関係数: {correlation:.2f}')
# データの抽出
x = df['中間１.HR'].values.reshape(-1, 1)  # 説明変数（心拍数）
y = df['中間1成績'].values  # 目的変数（成績）

# 線形回帰モデルの作成
model = LinearRegression()
model.fit(x, y)

# 回帰係数と切片を取得
slope = model.coef_[0]
intercept = model.intercept_

# 回帰直線の計算
y_pred = model.predict(x)
r_squared = model.score(x, y)
# 結果の表示
st.write(f"回帰係数: {slope}")
st.write(f"切片: {intercept}")
st.write(f"決定係数 (R^2): {r_squared:.2f}")
# 散布図と回帰直線のプロット
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='データポイント')
plt.plot(x, y_pred, color='red', label=f'回帰直線: y = {slope:.2f}x + {intercept:.2f}')
plt.title('中間1テスト時の心拍数と成績の回帰分析')
plt.xlabel('中間1テスト時の平均心拍数')
plt.ylabel('中間1成績')
plt.legend()
plt.grid(True)
st.pyplot(plt)

st.subheader('中間2結果')

x = df['中間２.HR']
y = df['中間2成績']
correlation = np.corrcoef(x, y)[0, 1]
st.write(f'相関係数: {correlation:.2f}')
# データの抽出
x = df['中間２.HR'].values.reshape(-1, 1)  # 説明変数（心拍数）
y = df['中間2成績'].values  # 目的変数（成績）

# 線形回帰モデルの作成
model = LinearRegression()
model.fit(x, y)

# 回帰係数と切片を取得
slope = model.coef_[0]
intercept = model.intercept_

# 回帰直線の計算
y_pred = model.predict(x)
r_squared = model.score(x, y)
# 結果の表示
st.write(f"回帰係数: {slope}")
st.write(f"切片: {intercept}")
st.write(f"決定係数 (R^2): {r_squared:.2f}")
# 散布図と回帰直線のプロット
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='データポイント')
plt.plot(x, y_pred, color='red', label=f'回帰直線: y = {slope:.2f}x + {intercept:.2f}')
plt.title('中間2テスト時の心拍数と成績の回帰分析')
plt.xlabel('中間2テスト時の平均心拍数')
plt.ylabel('中間2成績')
plt.legend()
plt.grid(True)
st.pyplot(plt)

st.subheader('最終成績結果')

x = df['最終.HR']
y = df['最終成績']
correlation = np.corrcoef(x, y)[0, 1]
st.write(f'相関係数: {correlation:.2f}')
# データの抽出
x = df['最終.HR'].values.reshape(-1, 1)  # 説明変数（心拍数）
y = df['最終成績'].values  # 目的変数（成績）

# 線形回帰モデルの作成
model = LinearRegression()
model.fit(x, y)

# 回帰係数と切片を取得
slope = model.coef_[0]
intercept = model.intercept_

# 回帰直線の計算
y_pred = model.predict(x)
r_squared = model.score(x, y)
# 結果の表示
st.write(f"回帰係数: {slope}")
st.write(f"切片: {intercept}")
st.write(f"決定係数 (R^2): {r_squared:.2f}")

# 散布図と回帰直線のプロット
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue', label='データポイント')
plt.plot(x, y_pred, color='red', label=f'回帰直線: y = {slope:.2f}x + {intercept:.2f}')
plt.title('最終テスト時の心拍数と成績の回帰分析')
plt.xlabel('最終テスト時の平均心拍数')
plt.ylabel('最終成績')
plt.legend()
plt.grid(True)
st.pyplot(plt)


st.subheader('分析結果からわかったこと')
st.write('中間1と中間2の試験には相関が見られなかったため、テスト中のストレス（心拍数）は成績に影響しない。'
         '中間試験ではプレッシャーが少ないからだと予測。'
         '最終試験での相関係数が0．44であり相関が少し見られる。最終試験において、ストレスと成績は影響している可能性がある。'
         '回帰係数は正なので、ストレスがかかるほど成績は伸びる。'
         '最後の頑張りが成績を向上させる。'
         'この分析では心拍数の個人差を考慮することはできていないのが問題点。')

st.write('データ源　→　A Wearable Exam Stress Dataset for Predicting Cognitive Performance in Real-World Settings　におけるHRのデータ　　https://physionet.org/content/wearable-exam-stress/1.0.0/')

st.subheader('重回帰分析を行う')

data = df = pd.read_csv('data-x.csv')
df = pd.DataFrame(data)


st.subheader('中間1結果')
# 説明変数と目的変数の設定
X = df[['中間1.平均HR', '中間1.平均EDA', '中間1.BVP']]
y = df['中間1成績（点）']

# 定数項を追加
X = sm.add_constant(X)

# 重回帰分析のモデル作成
model = sm.OLS(y, X).fit()

# 回帰係数、切片、決定係数の表示
st.write(f'回帰係数: {model.params[1:]}')
st.write(f'切片 (定数項): {model.params[0]}')
st.write(f'決定係数 (R²): {model.rsquared}')

# 残差のプロット
predictions = model.predict(X)
residuals = y - predictions

plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差プロット')
plt.xlabel('予測値')
plt.ylabel('残差')
st.pyplot(plt)

st.subheader('中間2結果')


X = df[['中間２.平HR', '中間2.平均EDA', '中間2.BVP']]
y = df['中間2成績']


# 定数項を追加
X = sm.add_constant(X)

# 重回帰分析のモデル作成
model = sm.OLS(y, X).fit()

# 回帰係数、切片、決定係数の表示
st.write(f'回帰係数: {model.params[1:]}')
st.write(f'切片 (定数項): {model.params[0]}')
st.write(f'決定係数 (R²): {model.rsquared}')

# 残差のプロット
predictions = model.predict(X)
residuals = y - predictions

plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差プロット')
plt.xlabel('予測値')
plt.ylabel('残差')
st.pyplot(plt)


st.subheader('最終試験結果')


X = df[['最終.平均HR', '最終.平均EDA', '最終.BVP']]
y = df['最終成績']


# 定数項を追加
X = sm.add_constant(X)

# 重回帰分析のモデル作成
model = sm.OLS(y, X).fit()

# 回帰係数、切片、決定係数の表示
st.write(f'回帰係数: {model.params[1:]}')
st.write(f'切片 (定数項): {model.params[0]}')
st.write(f'決定係数 (R²): {model.rsquared}')

# 残差のプロット
predictions = model.predict(X)
residuals = y - predictions

plt.figure(figsize=(8, 5))
plt.scatter(predictions, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差プロット')
plt.xlabel('予測値')
plt.ylabel('残差')
st.pyplot(plt)


st.write('・中間1の結果では、決定係数R²が非常に低く'
         '（0.00057）、このモデルはスコアを説明す'
         'る能力がほとんどないことがわかります。つま'
         'り、HRやEDA、BVPの変動は中間1のスコアにほ'
         'とんど影響を与えていない可能性')
st.write('・中間2の結果では、HRとEDA、BVPの'
         '全ての変数が正の回帰係数を持ってお'
         'り、これらが増加することでスコアが'
         '増加する傾向が示されています。しかし'
         '、決定係数R²が0.0457と依然として低'
         'いことから、これらの生理データがスコア'
         'に与える影響は限定的')
st.write('・最終試験の結果では、決定係数R²が0.324と'
         '中間試験に比べて大幅に高くなっており、この'
         'モデルがスコアの約32.4%を説明していることが'
         '示されています。HR、EDA、BVPの全てが正の回帰係数'
         'を持っており、これらの生理データの変動がスコアに影響'
         'を与えていることがわかります。特に、BVPの係数が'
         '非常に大きいことが特徴的で、この試験においてはBVPが'
         'スコアに強く影響を与えている可能性')
st.write('・中間1と中間2では、生理データのスコアに対する影響が小さい（R²が低い）ため、テストの状況やストレス、集中力など、他の要因がスコアに影響している可能性があります。'
          '最終試験では、生理データがスコアに対してより強い関連を示しており、特にBVPの影響が顕著です。これは、最終試験がよりプレッシャーの大きい状況で行われたため、心拍数や血流量の変化がパフォーマンスに影響を与えた可能性があります。')
