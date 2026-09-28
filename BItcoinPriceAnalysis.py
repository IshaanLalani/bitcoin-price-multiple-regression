import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

data = {
    'Trading_Volume': [18.2, 20.1, 22.1, 24.3, 26.8, 28.1, 30.2, 32.1, 34.4, 38.2, 41.0, 43.6, 40.1, 37.8, 35.2, 33.1, 31.4, 29.6, 27.8, 26.4, 23.1, 26.2, 29.1, 32.8, 36.1, 39.7, 42.4, 41.1, 43.3, 40.8],
    'Market_Cap': [1120, 1150, 1180, 1200, 1230, 1250, 1280, 1310, 1340, 1380, 1410, 1430, 1420, 1390, 1360, 1330, 1310, 1290, 1260, 1230, 1210, 1240, 1270, 1320, 1360, 1400, 1440, 1480, 1460, 1430],
    'Hash_Rate': [602, 606, 609, 611, 614, 617, 620, 623, 626, 629, 632, 635, 633, 630, 627, 624, 623, 621, 619, 616, 614, 617, 620, 624, 628, 631, 634, 638, 636, 633],
    'Circulating_Supply': [19.68, 19.68, 19.69, 19.69, 19.69, 19.70, 19.70, 19.70, 19.71, 19.71, 19.71, 19.72, 19.72, 19.72, 19.73, 19.73, 19.73, 19.74, 19.74, 19.74, 19.75, 19.75, 19.75, 19.76, 19.76, 19.76, 19.77, 19.77, 19.77, 19.78],
    'Bitcoin_Price': [55200, 56500, 57800, 59200, 60500, 61800, 63100, 64500, 65800, 67200, 68500, 69800, 70400, 68700, 67200, 61090, 64600, 63300, 62000, 60600, 59400, 61200, 62800, 64900, 67100, 69200, 70900, 73100, 72200, 70100]
}

df = pd.DataFrame(data)

scaler_cols = ['Trading_Volume', 'Market_Cap', 'Hash_Rate', 'Circulating_Supply']
df_normalized = df.copy()
for col in scaler_cols:
    df_normalized[col + '_norm'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

X = df[['Trading_Volume', 'Market_Cap', 'Circulating_Supply', 'Hash_Rate']]
y = df['Bitcoin_Price']

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm).fit()

print(model.summary())

vif_data = pd.DataFrame()
vif_data['Feature'] = X_sm.columns
vif_data['VIF'] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
print(vif_data)