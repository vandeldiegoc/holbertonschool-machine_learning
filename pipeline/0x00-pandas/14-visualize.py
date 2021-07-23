#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit='s')
df= df[df['Date'] >= '2017-01-01']
df = df.drop(columns=['Weighted_Price'])

df.fillna({'Volume_(BTC)': 0, 'Volume_(Currency)': 0}, inplace=True)
df['Close'].ffill(inplace=True)
df["Open"].fillna(df['Close'], inplace=True)
df["High"].fillna(value=df['Close'], inplace=True)
df["Low"].fillna(value=df['Close'], inplace=True)
df = df.set_index("Date")

df.plot()
plt.show()