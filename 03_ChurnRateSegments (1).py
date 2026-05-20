# ============================================================
# POWER BI PYTHON VISUAL SCRIPT
# Visual: Churn Rate by Key Segments (Contract, Internet, Payment, Offer)
# Dataset Required: Telco_Full_PowerBI_NLP.csv
# Columns needed: Contract, Internet Type, Payment Method, Offer,
#                 Churn_Predicted
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

df = dataset.copy()

BLUE = '#2980b9'
RED  = '#e74c3c'

# Compute churn rate % per segment
def churn_rate(df, col):
    return (
        df.groupby(col)['Churn_Predicted']
          .apply(lambda x: (x == 'Yes').mean() * 100)
          .sort_values(ascending=True)
    )

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#f9f9f9')
fig.suptitle('Churn Rate by Customer Segment', fontsize=16,
             fontweight='bold', y=1.01)

segments = [
    ('Contract',       'By Contract Type'),
    ('Internet Type',  'By Internet Type'),
    ('Payment Method', 'By Payment Method'),
    ('Offer',          'By Offer Type'),
]

for ax, (col, title) in zip(axes.flatten(), segments):
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if col not in df.columns:
        ax.text(0.5, 0.5, f'Column "{col}" not found',
                ha='center', va='center', transform=ax.transAxes, color='#e74c3c')
        ax.set_title(title, fontweight='bold', fontsize=12)
        continue

    rates  = churn_rate(df, col)
    norm   = (rates.values - rates.min()) / (rates.max() - rates.min() + 1e-9)
    colors = plt.cm.RdYlGn(1 - norm)     # red = high churn, green = low churn

    bars = ax.barh(rates.index, rates.values,
                   color=colors, edgecolor='white', height=0.65)

    for bar, val in zip(bars, rates.values):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=10, color='#2c3e50')

    ax.set_xlabel('Churn Rate (%)', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlim(right=rates.max() * 1.18)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
