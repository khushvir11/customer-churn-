# ============================================================
# POWER BI PYTHON VISUAL SCRIPT
# Visual: Single-Customer Real-Time Churn Prediction
#
# HOW TO USE:
#  1. In Power BI, create a table with exactly ONE row (use slicers
#     to filter down to one customer, or create a "What-If" parameter
#     table with the input columns listed below).
#  2. The script reads that row and shows the churn prediction.
#
# Dataset Required: Telco_Full_PowerBI_NLP.csv  (with all feature cols)
# Model file: catboost_churn_model.pkl  (update MODEL_PATH)
# Metadata:   model_metadata.pkl        (update META_PATH)
#
# Columns needed (passed by Power BI from the dataset):
#   All original feature columns used during training
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle

MODEL_PATH = r"C:\PowerBI_Telco\catboost_churn_model.pkl"   # <-- UPDATE
META_PATH  = r"C:\PowerBI_Telco\model_metadata.pkl"         # <-- UPDATE

df = dataset.copy()

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')

def gauge(ax, probability, title="Churn Risk Gauge"):
    """Draw a semicircular gauge for churn probability."""
    theta = np.linspace(np.pi, 0, 300)
    x_outer = np.cos(theta)
    y_outer = np.sin(theta)

    # Background arc segments (green → yellow → red)
    for i in range(len(theta) - 1):
        t = i / (len(theta) - 1)
        color = plt.cm.RdYlGn(1 - t)
        ax.fill_between([x_outer[i], x_outer[i+1]],
                        [y_outer[i]*0.5, y_outer[i+1]*0.5],
                        [y_outer[i]*1.0, y_outer[i+1]*1.0],
                        color=color, alpha=0.85)

    # Needle
    needle_angle = np.pi * (1 - probability / 100)
    nx = 0.75 * np.cos(needle_angle)
    ny = 0.75 * np.sin(needle_angle)
    ax.annotate('', xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='white',
                                lw=3, mutation_scale=20))
    centre = plt.Circle((0, 0), 0.07, color='white', zorder=10)
    ax.add_patch(centre)

    # Probability text
    color_text = '#e74c3c' if probability >= 60 else ('#f39c12' if probability >= 30 else '#27ae60')
    ax.text(0, -0.25, f'{probability:.1f}%',
            ha='center', va='center', fontsize=28,
            fontweight='bold', color=color_text)

    segment = ('🔴 HIGH RISK' if probability >= 60
               else ('🟡 MEDIUM RISK' if probability >= 30 else '🟢 LOW RISK'))
    ax.text(0, -0.45, segment, ha='center', va='center',
            fontsize=14, color='white', fontweight='bold')

    ax.text(-1.05, 0.0, '0%',   ha='center', color='#bdc3c7', fontsize=10)
    ax.text( 0,    1.05, '50%', ha='center', color='#bdc3c7', fontsize=10)
    ax.text( 1.05, 0.0, '100%', ha='center', color='#bdc3c7', fontsize=10)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=15, fontweight='bold',
                 color='white', pad=10)

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(META_PATH, 'rb') as f:
        meta = pickle.load(f)

    FEATURES = meta['features']
    CAT_FEATURES = meta['cat_features']
    CAT_IDX = meta['cat_feature_indices']

    # Use the first row of the filtered dataset
    row = df[FEATURES].head(1).copy()

    # Fill any missing categoricals
    for col in CAT_FEATURES:
        if col in row.columns:
            row[col] = row[col].fillna('Unknown').astype(str)

    from catboost import Pool
    pool = Pool(row, cat_features=CAT_IDX)
    prob = model.predict_proba(pool)[0, 1] * 100

    gauge(ax, prob, title="Customer Churn Risk")

    # Customer info strip
    cust_id = df['Customer ID'].iloc[0] if 'Customer ID' in df.columns else 'Selected Customer'
    contract = df['Contract'].iloc[0] if 'Contract' in df.columns else 'N/A'
    tenure   = df['Tenure in Months'].iloc[0] if 'Tenure in Months' in df.columns else 'N/A'
    fig.text(0.5, 0.02,
             f"Customer: {cust_id}  |  Contract: {contract}  |  Tenure: {tenure} months",
             ha='center', fontsize=9, color='#95a5a6')

except FileNotFoundError as e:
    ax.text(0.5, 0.5,
            f"File not found. Update MODEL_PATH / META_PATH.\n{e}",
            ha='center', va='center', transform=ax.transAxes,
            color='#e74c3c', fontsize=11)
except Exception as e:
    ax.text(0.5, 0.5, f"Error:\n{e}",
            ha='center', va='center', transform=ax.transAxes,
            color='#e74c3c', fontsize=11)

plt.tight_layout()
plt.show()
