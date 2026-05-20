# ============================================================
# POWER BI PYTHON VISUAL — SCRIPT 3
# Visual: Live Text Input → Category Prediction
#
# HOW TO SET UP THE TEXT INPUT IN POWER BI:
#   1. Modeling → New Parameter → Numeric Range
#      Name: "Churn Reason Index"  Min:0  Max:50  Step:1
#      (This lets users cycle through churn reasons with a slider)
#   2. Alternatively, use a slicer on "Customer ID" — drag ALL
#      columns into Python "Values" — the script picks row[0]
#
# FILES NEEDED:
#   nlp_label_encoder.pkl
#   nlp_tfidf_vectorizer.pkl
#   nlp_catboost_model.pkl
#
# POWER BI SETUP:
#   Dataset: Telco_Full_PowerBI_NLP.csv
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score,
#                       Customer ID
#   Slicer: Customer ID (single select)
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix

# ── UPDATE THIS PATH ─────────────────────────────────────────
PKL_FOLDER = r"C:\Users\khush\Downloads"
LE_PATH    = PKL_FOLDER + r"\nlp_label_encoder.pkl"
TF_PATH    = PKL_FOLDER + r"\nlp_tfidf_vectorizer.pkl"
CB_PATH    = PKL_FOLDER + r"\nlp_catboost_model.pkl"
# ─────────────────────────────────────────────────────────────

STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','have','has',
    'had','do','does','did','will','would','shall','should','may','might',
    'can','could','a','an','the','and','but','or','if','in','on','at',
    'to','for','of','with','by','from','as','about','up','out','not',
    'no','so','than','very','just','also','other','when','where','while'
}
POS_WORDS = {'good','great','excellent','best','better','happy','love',
             'nice','wonderful','satisfied','perfect','fine','improved'}
NEG_WORDS = {'bad','worst','terrible','horrible','awful','poor','hate',
             'disappointed','unhappy','rude','expensive','unreliable',
             'dropping','issues','problem','price','charges','cost',
             'high','overpriced','limited','slow','disconnecting'}

CAT_COLORS  = {
    'Competitor'     : '#e74c3c',
    'Price'          : '#e67e22',
    'Dissatisfaction': '#8e44ad',
    'Attitude'       : '#2980b9',
    'Other'          : '#27ae60',
}
CAT_ICONS = {
    'Competitor'     : '⚔  Competitor',
    'Price'          : '💰  Price',
    'Dissatisfaction': '😞  Dissatisfaction',
    'Attitude'       : '😠  Attitude',
    'Other'          : '📋  Other',
}
ACTIONS = {
    'Competitor'     : '→ Launch win-back offer; match competitor pricing.',
    'Price'          : '→ Offer loyalty discount or downgrade to cheaper plan.',
    'Dissatisfaction': '→ Escalate to service quality team; pro-active callback.',
    'Attitude'       : '→ Flag staff interaction; assign dedicated support rep.',
    'Other'          : '→ Review individually; tag for manual follow-up.',
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in re.sub(r'\s+', ' ', text).strip().split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

def simple_sentiment(text):
    tokens = text.lower().split()
    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    t = pos + neg + 1e-9
    c = (pos - neg) / t
    return c, pos/t, neg/t, max(0, 1-(pos+neg)/(len(tokens)+1e-9)), c, min(1.0,(pos+neg)/(len(tokens)+1e-9))

# ── Main ─────────────────────────────────────────────────────
import pandas as pd
df = dataset.copy()

fig = plt.figure(figsize=(14, 6))
fig.patch.set_facecolor('#0F0F1E')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax_gauge = fig.add_subplot(gs[0])
ax_bar   = fig.add_subplot(gs[1])
ax_info  = fig.add_subplot(gs[2])

for ax in [ax_gauge, ax_bar, ax_info]:
    ax.set_facecolor('#0F0F1E')
    ax.axis('off')

try:
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    with open(TF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(CB_PATH, 'rb') as f:
        model = pickle.load(f)

    row = df.iloc[0]
    churn_reason   = str(row.get('Churn Reason', 'No reason provided'))
    monthly_charge = float(row.get('Monthly Charge', 70))
    tenure         = float(row.get('Tenure in Months', 12))
    satisfaction   = float(row.get('Satisfaction Score', 2))
    cust_id        = str(row.get('Customer ID', 'N/A'))

    clean = clean_text(churn_reason)
    tf_vec = tfidf.transform([clean])
    compound, pos_s, neg_s, neu_s, pol, subj = simple_sentiment(churn_reason)
    meta = np.array([[
        len(churn_reason.split()), len(churn_reason),
        compound, neg_s, pos_s, neu_s, pol, subj,
        monthly_charge, tenure, satisfaction
    ]])
    combined = hstack([tf_vec, csr_matrix(meta)]).toarray()
    probs    = model.predict_proba(combined)[0]
    top3_idx = np.argsort(probs)[::-1][:3]
    top_cat  = le.classes_[top3_idx[0]]
    top_prob = probs[top3_idx[0]]
    top_col  = CAT_COLORS.get(top_cat, '#aaaaaa')

    # ── GAUGE ────────────────────────────────────────────────
    ax_gauge.axis('on')
    ax_gauge.set_xlim(-1.4, 1.4)
    ax_gauge.set_ylim(-1.0, 1.3)
    ax_gauge.set_aspect('equal')
    ax_gauge.axis('off')

    theta  = np.linspace(0, 2*np.pi, 300)
    filled = np.linspace(np.pi/2, np.pi/2 - top_prob*2*np.pi, 300)
    ax_gauge.plot(np.cos(theta), np.sin(theta),
                  color='#222244', linewidth=20, solid_capstyle='round')
    ax_gauge.plot(np.cos(filled), np.sin(filled),
                  color=top_col, linewidth=20, solid_capstyle='round')
    ax_gauge.text(0,  0.20, CAT_ICONS.get(top_cat,''), ha='center',
                  fontsize=20, color=top_col, fontweight='bold')
    ax_gauge.text(0, -0.10, f'{top_prob*100:.1f}%', ha='center',
                  fontsize=24, color='white', fontweight='bold')
    ax_gauge.text(0, -0.38, f'Customer: {cust_id}', ha='center',
                  fontsize=9, color='#888899')
    ax_gauge.set_title('Predicted Churn\nCategory', fontsize=12,
                        fontweight='bold', color='white', pad=10)

    # ── PROBABILITY BARS ─────────────────────────────────────
    ax_bar.axis('on')
    ax_bar.set_facecolor('#111128')
    all_cats  = le.classes_
    all_probs = probs
    order     = np.argsort(all_probs)
    b_colors  = [CAT_COLORS.get(le.classes_[i], '#aaaaaa') for i in order]
    bars = ax_bar.barh([le.classes_[i] for i in order],
                        [all_probs[i] for i in order],
                        color=b_colors, edgecolor='#0F0F1E',
                        height=0.55)
    for bar, i in zip(bars, order):
        ax_bar.text(bar.get_width() + 0.005,
                    bar.get_y() + bar.get_height()/2,
                    f'{all_probs[i]*100:.1f}%',
                    va='center', fontsize=10, color='white')
    ax_bar.set_xlim(0, 0.55)
    ax_bar.set_xlabel('Probability', fontsize=9, color='#AAAAAA')
    ax_bar.set_title('All Category\nProbabilities', fontsize=12,
                      fontweight='bold', color='white', pad=10)
    ax_bar.tick_params(colors='#CCCCCC', labelsize=9)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color('#333355')
    ax_bar.spines['bottom'].set_color('#333355')
    ax_bar.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax_bar.grid(True, axis='x', alpha=0.12, color='white')

    # ── ACTION PANEL ─────────────────────────────────────────
    ax_info.axis('off')
    ax_info.set_title('Retention Action', fontsize=12,
                       fontweight='bold', color='white', pad=10)

    reason_short = (churn_reason[:70]+'…') if len(churn_reason)>70 else churn_reason
    sent_label = ('Negative' if compound<-0.05 else
                  'Neutral'  if compound<0.05  else 'Positive')
    sent_col   = ('#e74c3c' if compound<-0.05 else
                  '#f39c12' if compound<0.05  else '#27ae60')

    info_lines = [
        ('Churn Reason:', reason_short, '#CCCCCC', 9, 'normal'),
        ('', '', '#CCCCCC', 6, 'normal'),
        ('Sentiment:', f'{sent_label}  ({compound:+.2f})', sent_col, 10, 'bold'),
        ('', '', '#CCCCCC', 6, 'normal'),
        ('Monthly Charge:', f'${monthly_charge:.2f}', '#f39c12', 10, 'bold'),
        ('Tenure:', f'{int(tenure)} months', '#3498db', 10, 'bold'),
        ('Satisfaction:', f'{int(satisfaction)} / 5', '#27ae60', 10, 'bold'),
        ('', '', '#CCCCCC', 8, 'normal'),
        ('Recommended Action:', '', '#FFFFFF', 10, 'bold'),
        ('', ACTIONS.get(top_cat, '→ Review manually.'), top_col, 10, 'normal'),
    ]

    y = 0.95
    for label, value, color, size, weight in info_lines:
        if label and value:
            ax_info.text(0.02, y, label, transform=ax_info.transAxes,
                         fontsize=size-1, color='#888899', va='top')
            ax_info.text(0.02, y - 0.042, value,
                         transform=ax_info.transAxes,
                         fontsize=size, color=color, va='top',
                         fontweight=weight, wrap=True)
            y -= 0.11
        elif value:
            ax_info.text(0.02, y, value, transform=ax_info.transAxes,
                         fontsize=size, color=color, va='top',
                         fontweight=weight, wrap=True)
            y -= 0.10
        else:
            y -= 0.01

except FileNotFoundError as e:
    for ax in [ax_gauge, ax_bar, ax_info]:
        ax.text(0.5, 0.5,
                f'Update PKL_FOLDER:\n{PKL_FOLDER}\n\n{e}',
                ha='center', va='center', transform=ax.transAxes,
                color='#e74c3c', fontsize=9)
except Exception as e:
    ax_gauge.text(0.5, 0.5, f'Error:\n{e}',
                  ha='center', va='center', transform=ax_gauge.transAxes,
                  color='#e74c3c', fontsize=9)

plt.tight_layout()
plt.show()
