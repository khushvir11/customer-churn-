# ============================================================
# POWER BI PYTHON VISUAL SCRIPT
# Visual: NLP Churn Reason Category Analysis + Sentiment
# Dataset Required: Telco_Full_PowerBI_NLP.csv
# Columns needed: NLP_Predicted_Category, NLP_Sentiment_Score,
#                 NLP_Sentiment_Label, Churn_Predicted, Monthly Charge
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

df = dataset.copy()

# Only churned customers have NLP columns
churned = df[df['NLP_Predicted_Category'].notna() &
             (df['NLP_Predicted_Category'] != 'N/A')].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#f9f9f9')
fig.suptitle('NLP Analysis — Churn Reason & Sentiment',
             fontsize=16, fontweight='bold')

# ── Plot 1: Churn Category Frequency ──────────────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

if 'NLP_Predicted_Category' in churned.columns and len(churned) > 0:
    cat_counts = churned['NLP_Predicted_Category'].value_counts().sort_values()
    colors = plt.cm.Set2(np.linspace(0, 1, len(cat_counts)))
    bars = ax.barh(cat_counts.index, cat_counts.values,
                   color=colors, edgecolor='white', height=0.65)
    for bar, val in zip(bars, cat_counts.values):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                str(val), va='center', fontsize=10)
    ax.set_title('Churn Category (NLP-Predicted)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Number of Customers')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
else:
    ax.text(0.5, 0.5, 'No NLP data available', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c')

# ── Plot 2: Avg Sentiment Score by Category ────────────────────
ax = axes[1]
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

if ('NLP_Predicted_Category' in churned.columns and
        'NLP_Sentiment_Score' in churned.columns and len(churned) > 0):
    sent_avg = (churned.groupby('NLP_Predicted_Category')['NLP_Sentiment_Score']
                       .mean()
                       .sort_values())
    bar_colors = ['#e74c3c' if v < -0.05 else ('#f39c12' if v < 0.05 else '#27ae60')
                  for v in sent_avg.values]
    bars = ax.barh(sent_avg.index, sent_avg.values,
                   color=bar_colors, edgecolor='white', height=0.65)
    ax.axvline(0, color='#2c3e50', lw=1.5, linestyle='--')
    for bar, val in zip(bars, sent_avg.values):
        x_pos = bar.get_width() + 0.005 if val >= 0 else bar.get_width() - 0.005
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9)
    ax.set_title('Avg Sentiment Score by Category', fontweight='bold', fontsize=12)
    ax.set_xlabel('Avg VADER Sentiment Score (-1 to +1)')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
else:
    ax.text(0.5, 0.5, 'No sentiment data', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c')

# ── Plot 3: Revenue at Risk by NLP Category ───────────────────
ax = axes[2]
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

charge_col = 'Monthly Charge' if 'Monthly Charge' in churned.columns else None
if ('NLP_Predicted_Category' in churned.columns and
        charge_col and len(churned) > 0):
    rev_risk = (churned.groupby('NLP_Predicted_Category')[charge_col]
                       .sum()
                       .sort_values())
    colors = plt.cm.Reds(np.linspace(0.3, 0.85, len(rev_risk)))
    bars = ax.barh(rev_risk.index, rev_risk.values,
                   color=colors, edgecolor='white', height=0.65)
    for bar, val in zip(bars, rev_risk.values):
        ax.text(bar.get_width() + 50,
                bar.get_y() + bar.get_height() / 2,
                f'${val:,.0f}', va='center', fontsize=9)
    ax.set_title('Monthly Revenue at Risk by Category', fontweight='bold', fontsize=12)
    ax.set_xlabel('Total Monthly Revenue at Risk ($)')
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
else:
    ax.text(0.5, 0.5, 'No revenue data', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c')

plt.tight_layout()
plt.show()
