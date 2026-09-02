import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib


# 元のデータ
df = pd.read_csv("MedicalReportPubH_2024.csv", encoding="shift_jis")
print(df["発生要因_当事者の行動に関わる要因"].value_counts().head(20))

# 発生曜日
df_day = df["発生曜日"].value_counts()

plt.figure(figsize=(8, 5))
df_day.plot(kind="bar")
plt.title("Near misses by day of the week")
plt.xlabel("day")
plt.ylabel("counts")

# 発生時間
df_delete = (
    df["発生時間帯"]
    [df["発生時間帯"] != "不明"]
    .str.extract(r"(\d+)")
    .astype(int)
)

print(df_delete.head())

