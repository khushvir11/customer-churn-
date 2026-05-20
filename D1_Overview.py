# ╔══════════════════════════════════════════════════════════════╗
# ║  DASHBOARD 1 — OVERVIEW  (Power BI Python Visual Script)    ║
# ║  Visuals:                                                    ║
# ║   1. KPI Cards : Total | Churned | Churn Rate | Avg Charge  ║
# ║   2. Pie       : Churn Distribution                         ║
# ║   3. Bar       : Churn by Gender                            ║
# ║   4. Column    : Churn by Senior Citizen                    ║
# ║   5. Donut     : Contract Type Distribution                 ║
# ║   + PREDICTIVE : Predicted Churn Risk overlay on each card  ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  POWER BI SETUP STEPS                                       ║
# ║  1. Get Data → CSV → load telco dataset                     ║
# ║  2. Insert → Python Visual                                  ║
# ║  3. Drag columns into Values:                               ║
# ║       Gender, Senior Citizen, Contract,                     ║
# ║       Monthly Charge, Churn Label,                          ║
# ║       Tenure in Months, Total Charges                       ║
# ║  4. Add Slicers (optional) for Contract, Gender,            ║
# ║       Senior Citizen — visuals will filter automatically    ║
# ║  5. Paste this script → click Run                           ║
# ║  6. Update RF_PATH / SC_PATH to your saved model files      ║
# ╚══════════════════════════════════════════════════════════════╝

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import joblib
import warnings
warnings.filterwarnings('ignore')

# ── MODEL PATHS — update to your local folder ─────────────────



RF_PATH = r"C:\Users\khush\Downloads\nlp_catboost_model.pkl"

SC_PATH = r"C:\Users\khush\Downloads\nlp_label_encoder.pkl"
# ──────────────────────────────────────────────────────────────

from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════════════════
#  LOAD & PREPROCESS  (mirrors your notebook exactly)
# ══════════════════════════════════════════════════════════════
df_raw = dataset.copy()
df_raw.columns = df_raw.columns.str.strip()

# ── Types ─────────────────────────────────────────────────────
for col in ['Churn Label', 'Senior Citizen']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0).astype(int)
for col in ['Monthly Charge', 'Tenure in Months', 'Total Charges']:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

# ── KPIs (actual) ────────────────────────────────────────────
total_cust   = len(df_raw)
actual_churn = int(df_raw['Churn Label'].sum()) if 'Churn Label' in df_raw.columns else 0
actual_rate  = actual_churn / total_cust * 100 if total_cust > 0 else 0
avg_charge   = df_raw['Monthly Charge'].mean() if 'Monthly Charge' in df_raw.columns else 0

# ══════════════════════════════════════════════════════════════
#  PREDICTIVE LAYER — Random Forest scoring on filtered data
# ══════════════════════════════════════════════════════════════
try:
    rf     = joblib.load(RF_PATH)
    scaler = joblib.load(SC_PATH)

    drop_cols = ['Customer ID', 'Country', 'State', 'City', 'Zip Code',
                 'Latitude', 'Longitude', 'Quarter', 'Churn Category',
                 'Churn Reason', 'Customer Status', 'Churn Label']

    df_pred = df_raw.copy()
    df_pred = df_pred.drop(columns=[c for c in drop_cols if c in df_pred.columns])

    le = LabelEncoder()
    for col in df_pred.select_dtypes(include='object').columns:
        df_pred[col] = le.fit_transform(df_pred[col].astype(str))
    df_pred = df_pred.fillna(0)

    # Align columns with what RF was trained on
    expected = rf.feature_names_in_ if hasattr(rf, 'feature_names_in_') else df_pred.columns
    for c in expected:
        if c not in df_pred.columns:
            df_pred[c] = 0
    df_pred = df_pred[expected]

    pred_probs       = rf.predict_proba(df_pred)[:, 1]
    predicted_churn  = int((pred_probs >= 0.5).sum())
    predicted_rate   = predicted_churn / total_cust * 100
    avg_risk         = pred_probs.mean() * 100
    model_loaded     = True
except Exception:
    predicted_churn = actual_churn
    predicted_rate  = actual_rate
    avg_risk        = actual_rate
    model_loaded    = False

# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════════════════════
RED    = '#E74C3C'
GREEN  = '#27AE60'
BLUE   = '#2980B9'
PURPLE = '#8E44AD'
ORANGE = '#E67E22'
TEAL   = '#16A085'
GREY   = '#95A5A6'
BG     = '#F0F3F7'
CARD_BG= '#FFFFFF'

# ══════════════════════════════════════════════════════════════
#  FIGURE LAYOUT  3 rows × 4 cols
# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(3, 4, figure=fig,
                        hspace=0.55, wspace=0.38,
                        height_ratios=[0.22, 0.40, 0.38])

# ══════════════════════════════════════════════════════════════
#  ROW 0 — KPI CARDS  (actual value + predicted below)
# ══════════════════════════════════════════════════════════════
card_specs = [
    ('Total Customers',       f'{total_cust:,}',
     f'Predicted at risk: {predicted_churn:,}', BLUE),
    ('Churned Customers',     f'{actual_churn:,}',
     f'Model predicts: {predicted_churn:,}',    RED),
    ('Churn Rate',            f'{actual_rate:.1f}%',
     f'Predicted rate: {predicted_rate:.1f}%',  ORANGE),
    ('Avg Monthly Charge',    f'${avg_charge:.2f}',
     f'Avg risk score: {avg_risk:.1f}%',        GREEN),
]
for i, (title, val, pred_line, color) in enumerate(card_specs):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(color)
        sp.set_linewidth(3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.5, 0.72, val, ha='center', va='center',
            fontsize=22, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.40, title, ha='center', va='center',
            fontsize=9, color='#555555', transform=ax.transAxes)
    ax.axhline(y=0.22, color='#EEEEEE', linewidth=1,
               xmin=0.05, xmax=0.95)
    model_tag = '🤖 ' if model_loaded else '📊 '
    ax.text(0.5, 0.12, model_tag + pred_line, ha='center', va='center',
            fontsize=7.5, color=color, style='italic',
            transform=ax.transAxes)

# ══════════════════════════════════════════════════════════════
#  ROW 1 — PIE (cols 0-1)  |  BAR gender (col 2)  |  COL senior (col 3)
# ══════════════════════════════════════════════════════════════

# ── PIE: Churn Distribution — Actual vs Predicted ────────────
ax_pie = fig.add_subplot(gs[1, 0:2])
ax_pie.set_facecolor(BG)

stayed_act  = total_cust - actual_churn
stayed_pred = total_cust - predicted_churn

categories = ['Actual Churned', 'Actual Stayed',
              'Predicted Churn', 'Predicted Stayed']
sizes = [actual_churn, stayed_act, predicted_churn, stayed_pred]
colors_p = [RED, GREEN, '#FF8C69', '#52D68A']

# Only show actual in pie; add predicted as annotation
wedges, texts, autos = ax_pie.pie(
    [actual_churn, stayed_act],
    labels=[f'Churned\n{actual_churn:,}', f'Retained\n{stayed_act:,}'],
    colors=[RED, GREEN],
    autopct='%1.1f%%' , startangle=140,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2.5},
    textprops={'fontsize': 10}
)
for at in autos:
    at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
    
pred_patch = mpatches.Patch(color='#FF8C69',
    label=f'Predicted churn: {predicted_churn:,} ({predicted_rate:.1f}%)')
ax_pie.legend(handles=[pred_patch], loc='lower center',
              bbox_to_anchor=(0.5, -0.12), fontsize=9)
ax_pie.set_title('Churn Distribution\n(Actual vs Model Predicted)',
                 fontsize=12, fontweight='bold', color='#333333', pad=8)

# ── BAR: Churn by Gender ──────────────────────────────────────
ax_gen = fig.add_subplot(gs[1, 2])
ax_gen.set_facecolor(BG)
if 'Gender' in df_raw.columns:
    gen_grp = df_raw.groupby('Gender')['Churn Label'].agg(['sum', 'count'])
    gen_grp['rate'] = gen_grp['sum'] / gen_grp['count'] * 100
    x_pos = np.arange(len(gen_grp))
    bars_act = ax_gen.bar(x_pos - 0.2, gen_grp['rate'],
                          width=0.35, color=RED,
                          label='Actual', edgecolor='white')
    # Predicted per gender
    if model_loaded:
        df_raw['pred_prob'] = pred_probs
        gen_pred = df_raw.groupby('Gender')['pred_prob'].mean() * 100
        bars_prd = ax_gen.bar(x_pos + 0.2, gen_pred.values,
                              width=0.35, color='#FF8C69',
                              label='Predicted', edgecolor='white')
        for bar, v in zip(bars_prd, gen_pred.values):
            ax_gen.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5, f'{v:.1f}%',
                        ha='center', fontsize=8, color='#333333')
    for bar, v in zip(bars_act, gen_grp['rate']):
        ax_gen.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5, f'{v:.1f}%',
                    ha='center', fontsize=8, fontweight='bold', color='#333333')
    ax_gen.set_xticks(x_pos)
    ax_gen.set_xticklabels(gen_grp.index, fontsize=9)
    ax_gen.legend(fontsize=8)
ax_gen.set_title('Churn by Gender\n(Actual vs Predicted)', fontsize=11,
                 fontweight='bold', color='#333333')
ax_gen.set_ylabel('Churn Rate (%)', fontsize=9, color='#555555')
ax_gen.spines[['top', 'right']].set_visible(False)
ax_gen.tick_params(colors='#555555')

# ── COLUMN: Churn by Senior Citizen ──────────────────────────
ax_sen = fig.add_subplot(gs[1, 3])
ax_sen.set_facecolor(BG)
if 'Senior Citizen' in df_raw.columns:
    df_raw['SC_Label'] = df_raw['Senior Citizen'].map({0: 'Non-Senior', 1: 'Senior'})
    sc_grp = df_raw.groupby('SC_Label')['Churn Label'].agg(['sum', 'count'])
    sc_grp['rate'] = sc_grp['sum'] / sc_grp['count'] * 100
    x_pos2 = np.arange(len(sc_grp))
    bars_s = ax_sen.bar(x_pos2 - 0.2, sc_grp['rate'],
                        width=0.35, color=[GREEN, RED],
                        edgecolor='white', label='Actual')
    if model_loaded:
        sc_pred = df_raw.groupby('SC_Label')['pred_prob'].mean() * 100
        ax_sen.bar(x_pos2 + 0.2, sc_pred.values,
                   width=0.35, color=['#52D68A', '#FF8C69'],
                   edgecolor='white', label='Predicted')
        for i, v in enumerate(sc_pred.values):
            ax_sen.text(x_pos2[i] + 0.2, v + 0.5, f'{v:.1f}%',
                        ha='center', fontsize=8, color='#333333')
    for bar, v in zip(bars_s, sc_grp['rate']):
        ax_sen.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5, f'{v:.1f}%',
                    ha='center', fontsize=8, fontweight='bold', color='#333333')
    ax_sen.set_xticks(x_pos2)
    ax_sen.set_xticklabels(sc_grp.index, fontsize=9)
    ax_sen.legend(fontsize=8)
ax_sen.set_title('Churn by Senior Citizen\n(Actual vs Predicted)', fontsize=11,
                 fontweight='bold', color='#333333')
ax_sen.set_ylabel('Churn Rate (%)', fontsize=9, color='#555555')
ax_sen.spines[['top', 'right']].set_visible(False)
ax_sen.tick_params(colors='#555555')

# ══════════════════════════════════════════════════════════════
#  ROW 2 — DONUT: Contract Type (cols 0-1) | Predicted Risk (2-3)
# ══════════════════════════════════════════════════════════════

# ── DONUT: Contract Type Distribution ────────────────────────
ax_donut = fig.add_subplot(gs[2, 0:2])
ax_donut.set_facecolor(BG)
if 'Contract' in df_raw.columns:
    ct_counts = df_raw['Contract'].value_counts()
    d_colors  = [BLUE, ORANGE, TEAL][:len(ct_counts)]
    wedges_d, texts_d, autos_d = ax_donut.pie(
        ct_counts.values,
        labels=ct_counts.index,
        colors=d_colors,
        autopct='%1.1f%%', startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2.5, 'width': 0.58},
        textprops={'fontsize': 10}
    )
    for at in autos_d:
        at.set_fontsize(10); at.set_fontweight('bold'); at.set_color('white')
    ax_donut.text(0, 0, f'{total_cust:,}\nCustomers',
                  ha='center', va='center',
                  fontsize=11, fontweight='bold', color='#333333')
ax_donut.set_title('Contract Type Distribution', fontsize=12,
                   fontweight='bold', color='#333333', pad=10)

# ── PREDICTED RISK by Contract ────────────────────────────────
ax_risk = fig.add_subplot(gs[2, 2:4])
ax_risk.set_facecolor(BG)
if 'Contract' in df_raw.columns:
    ct_actual = df_raw.groupby('Contract')['Churn Label'].mean() * 100
    ct_actual = ct_actual.sort_values(ascending=True)
    x_c = np.arange(len(ct_actual))

    b_act = ax_risk.bar(x_c - 0.2, ct_actual.values,
                        width=0.35, color=RED,
                        label='Actual Churn %', edgecolor='white')
    if model_loaded:
        ct_pred = df_raw.groupby('Contract')['pred_prob'].mean() * 100
        ct_pred = ct_pred.reindex(ct_actual.index)
        b_prd = ax_risk.bar(x_c + 0.2, ct_pred.values,
                            width=0.35, color='#FF8C69',
                            label='Predicted Churn %', edgecolor='white')
        for bar, v in zip(b_prd, ct_pred.values):
            ax_risk.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 0.4, f'{v:.1f}%',
                         ha='center', fontsize=9, color='#333333')
    for bar, v in zip(b_act, ct_actual.values):
        ax_risk.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.4, f'{v:.1f}%',
                     ha='center', fontsize=9,
                     fontweight='bold', color='#333333')
    ax_risk.set_xticks(x_c)
    ax_risk.set_xticklabels(ct_actual.index, fontsize=9)
    ax_risk.legend(fontsize=9)
ax_risk.set_title('Churn Rate by Contract Type\n(Actual vs Predicted)',
                  fontsize=11, fontweight='bold', color='#333333')
ax_risk.set_ylabel('Churn Rate (%)', fontsize=9, color='#555555')
ax_risk.spines[['top', 'right']].set_visible(False)
ax_risk.tick_params(colors='#555555')

fig.suptitle('Dashboard 1 — Customer Churn Overview  |  Actual + Predictive Layer',
             fontsize=14, fontweight='bold', color='#1A1A2E', y=1.01)

plt.savefig('D1_Overview.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.show()
