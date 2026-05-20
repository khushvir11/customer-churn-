# ============================================================
# POWER BI PYTHON VISUAL SCRIPT
# Visual: Confusion Matrix + ROC Curve (Model Performance)
# Dataset Required: Telco_Full_PowerBI_NLP.csv
# Columns needed: Churn Label, Churn_Predicted, Churn_Probability,
#                 Prediction_Correct
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

df = dataset.copy()

# Prepare labels
df['_actual']   = df['Churn Label'].map({'Yes': 1, 'No': 0})
df['_predicted'] = df['Churn_Predicted'].map({'Yes': 1, 'No': 0})
df['_prob']     = df['Churn_Probability'] / 100.0

# Drop rows with nulls in key cols
df = df.dropna(subset=['_actual', '_predicted', '_prob'])

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('#f9f9f9')
fig.suptitle('CatBoost Model Performance', fontsize=16, fontweight='bold')

# ── 1. Confusion Matrix ────────────────────────────────────────
ax = axes[0]
ax.set_facecolor('#ffffff')
cm = confusion_matrix(df['_actual'], df['_predicted'])
labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
counts = [f'{v:,}' for v in cm.flatten()]
pcts   = [f'{v:.1%}' for v in cm.flatten() / cm.sum()]
annots = [f'{l}\n{c}\n({p})'
          for l, c, p in zip(labels, counts, pcts)]
annots = np.array(annots).reshape(2, 2)
sns.heatmap(cm, annot=annots, fmt='', cmap='Blues', ax=ax,
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'],
            linewidths=1.5, linecolor='white',
            annot_kws={'size': 10})
ax.set_title('Confusion Matrix', fontweight='bold', fontsize=13)
ax.set_ylabel('Actual', fontsize=11)
ax.set_xlabel('Predicted', fontsize=11)

# ── 2. ROC Curve ──────────────────────────────────────────────
ax = axes[1]
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
try:
    fpr, tpr, _ = roc_curve(df['_actual'], df['_prob'])
    auc = roc_auc_score(df['_actual'], df['_prob'])
    ax.plot(fpr, tpr, color='#e74c3c', lw=2.5,
            label=f'CatBoost  AUC = {auc:.4f}')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#e74c3c')
except Exception:
    auc = 0.0
ax.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--', lw=1.5,
        label='Random')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curve', fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(True, alpha=0.3, linestyle='--')

# ── 3. Metrics Summary ────────────────────────────────────────
ax = axes[2]
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

from sklearn.metrics import (accuracy_score, f1_score,
                              precision_score, recall_score)
try:
    metrics = {
        'Accuracy':  accuracy_score(df['_actual'], df['_predicted']) * 100,
        'ROC-AUC':   roc_auc_score(df['_actual'], df['_prob']) * 100,
        'F1 Score':  f1_score(df['_actual'], df['_predicted']) * 100,
        'Precision': precision_score(df['_actual'], df['_predicted'],
                                     zero_division=0) * 100,
        'Recall':    recall_score(df['_actual'], df['_predicted'],
                                  zero_division=0) * 100,
    }
except Exception as e:
    metrics = {'Error': 0}

bar_colors = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#e74c3c']
bars = ax.bar(list(metrics.keys()), list(metrics.values()),
              color=bar_colors[:len(metrics)],
              edgecolor='white', width=0.55, zorder=3)
ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Model Metrics (Test Set)', fontweight='bold', fontsize=13)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.axhline(100, color='#bdc3c7', linestyle='--', lw=1)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')

for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{val:.1f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold', color='#2c3e50')

ax.tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()
