# ============================================================
# POWER BI PYTHON VISUAL — SCRIPT 1
# Visual: NLP Churn Category Prediction (Gauge + Top-3 Bar)
#
# FILES NEEDED — place all 3 in the same folder:
#   nlp_label_encoder.pkl
#   nlp_tfidf_vectorizer.pkl
#   nlp_catboost_model.pkl
#
# UPDATE the folder path below (PKL_FOLDER) before using.
#
# POWER BI SETUP:
#   Dataset: Telco_Full_PowerBI_NLP.csv
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score,
#                       NLP_Predicted_Category
#   Use a slicer on Customer ID to pick one customer at a time.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import hstack, csr_matrix

# ── UPDATE THIS PATH ─────────────────────────────────────────
PKL_FOLDER = r"C:\PowerBI_Telco"
LE_PATH    = PKL_FOLDER + r"\nlp_label_encoder.pkl"
TF_PATH    = PKL_FOLDER + r"\nlp_tfidf_vectorizer.pkl"
CB_PATH    = PKL_FOLDER + r"\nlp_catboost_model.pkl"
# ─────────────────────────────────────────────────────────────

# ── Minimal English stopwords (no NLTK download required) ────
STOP_WORDS = {
    'i','me','my','we','our','you','your','he','him','his','she','her',
    'it','its','they','them','their','what','which','who','this','that',
    'these','those','am','is','are','was','were','be','been','have','has',
    'had','do','does','did','will','would','shall','should','may','might',
    'can','could','a','an','the','and','but','or','if','in','on','at',
    'to','for','of','with','by','from','as','about','up','out','not',
    'no','so','than','very','just','also','other','when','where','while'
}

# ── Simple sentiment scorer (no NLTK vader required) ─────────
POS_WORDS = {'good','great','excellent','best','better','happy','love',
             'nice','wonderful','satisfied','perfect','fine','improved'}
NEG_WORDS = {'bad','worst','terrible','horrible','awful','poor','hate',
             'disappointed','unhappy','rude','expensive','unreliable',
             'dropping','issues','problem','price','charges','cost',
             'high','overpriced','limited','slow','disconnecting'}

def simple_sentiment(text):
    tokens = text.lower().split()
    pos = sum(1 for w in tokens if w in POS_WORDS)
    neg = sum(1 for w in tokens if w in NEG_WORDS)
    total = pos + neg + 1e-9
    compound    = (pos - neg) / total
    pos_score   = pos / total
    neg_score   = neg / total
    neu_score   = max(0, 1 - (pos + neg) / (len(tokens) + 1e-9))
    polarity    = compound
    subjectivity= min(1.0, (pos + neg) / (len(tokens) + 1e-9))
    return compound, pos_score, neg_score, neu_score, polarity, subjectivity

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = [w for w in re.sub(r'\s+', ' ', text).strip().split()
              if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(tokens)

# ── Category display config ───────────────────────────────────
CAT_COLORS = {
    'Competitor'     : '#e74c3c',
    'Price'          : '#e67e22',
    'Dissatisfaction': '#8e44ad',
    'Attitude'       : '#2980b9',
    'Other'          : '#27ae60',
}
CAT_ICONS = {
    'Competitor'     : '⚔',
    'Price'          : '💰',
    'Dissatisfaction': '😞',
    'Attitude'       : '😠',
    'Other'          : '📋',
}

# ── Main ─────────────────────────────────────────────────────
df = dataset.copy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor('#1C1C2E')

for ax in axes:
    ax.set_facecolor('#1C1C2E')
    ax.tick_params(colors='#CCCCCC')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

try:
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    with open(TF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(CB_PATH, 'rb') as f:
        model = pickle.load(f)

    # Get data from the (slicer-filtered) dataset
    churn_reason = str(df['Churn Reason'].iloc[0]) \
        if 'Churn Reason' in df.columns else 'No reason provided'
    monthly_charge = float(df['Monthly Charge'].iloc[0]) \
        if 'Monthly Charge' in df.columns else 70.0
    tenure = float(df['Tenure in Months'].iloc[0]) \
        if 'Tenure in Months' in df.columns else 12.0
    satisfaction = float(df['Satisfaction Score'].iloc[0]) \
        if 'Satisfaction Score' in df.columns else 2.0

    # NLP pipeline
    clean = clean_text(churn_reason)
    tfidf_vec = tfidf.transform([clean])
    compound, pos_s, neg_s, neu_s, polarity, subjectivity = simple_sentiment(churn_reason)

    meta = np.array([[
        len(churn_reason.split()), len(churn_reason),
        compound, neg_s, pos_s, neu_s,
        polarity, subjectivity,
        monthly_charge, tenure, satisfaction
    ]])
    combined = hstack([tfidf_vec, csr_matrix(meta)]).toarray()
    probs    = model.predict_proba(combined)[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    top_cat  = le.classes_[top3_idx[0]]
    top_prob = probs[top3_idx[0]]
    top_color= CAT_COLORS.get(top_cat, '#aaaaaa')
    top_icon = CAT_ICONS.get(top_cat, '📋')

    # ── LEFT: Prediction card ─────────────────────────────────
    ax1 = axes[0]
    ax1.axis('off')

    # Outer ring
    theta = np.linspace(0, 2 * np.pi, 300)
    ax1.plot(np.cos(theta) * 1.0, np.sin(theta) * 1.0,
             color='#333355', linewidth=18, solid_capstyle='round')
    # Filled arc proportional to confidence
    filled = np.linspace(np.pi / 2, np.pi / 2 - top_prob * 2 * np.pi, 300)
    ax1.plot(np.cos(filled), np.sin(filled),
             color=top_color, linewidth=18, solid_capstyle='round')

    ax1.text(0,  0.22, f'{top_icon}', ha='center', va='center',
             fontsize=26, color=top_color)
    ax1.text(0, -0.05, top_cat, ha='center', va='center',
             fontsize=18, fontweight='bold', color='white')
    ax1.text(0, -0.30, f'{top_prob * 100:.1f}% confidence',
             ha='center', va='center', fontsize=12, color='#AAAAAA')

    # Sentiment badge
    sent_color = ('#e74c3c' if compound < -0.05 else
                  '#f39c12' if compound < 0.05 else '#27ae60')
    sent_label = ('Negative' if compound < -0.05 else
                  'Neutral'  if compound < 0.05 else 'Positive')
    ax1.text(0, -0.60,
             f'Sentiment: {sent_label}  ({compound:+.2f})',
             ha='center', va='center', fontsize=10, color=sent_color)

    # Churn reason text (wrapped)
    reason_display = (churn_reason[:55] + '…') \
        if len(churn_reason) > 55 else churn_reason
    ax1.text(0, -0.82, f'"{reason_display}"',
             ha='center', va='center', fontsize=8.5,
             color='#888899', style='italic')

    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.1, 1.2)
    ax1.set_title('Predicted Churn Category', fontsize=13,
                  fontweight='bold', color='white', pad=14)

    # ── RIGHT: Top-3 probabilities ────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1C1C2E')

    cat_names  = [le.classes_[i] for i in top3_idx]
    cat_probs  = [probs[i]       for i in top3_idx]
    bar_colors = [CAT_COLORS.get(c, '#aaaaaa') for c in cat_names]

    y_pos = np.arange(len(cat_names))
    bars  = ax2.barh(y_pos, cat_probs,
                     color=bar_colors, edgecolor='#1C1C2E',
                     height=0.55, linewidth=1.5)

    for bar, prob, cat in zip(bars, cat_probs, cat_names):
        icon = CAT_ICONS.get(cat, '')
        ax2.text(bar.get_width() + 0.008,
                 bar.get_y() + bar.get_height() / 2,
                 f'{prob * 100:.1f}%  {icon}',
                 va='center', fontsize=12, color='white', fontweight='bold')

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(cat_names, fontsize=12, color='white')
    ax2.set_xlim(0, 0.60)
    ax2.set_xlabel('Probability', fontsize=10, color='#AAAAAA')
    ax2.set_title('Top-3 Category Predictions', fontsize=13,
                  fontweight='bold', color='white', pad=14)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333355')
    ax2.spines['bottom'].set_color('#333355')
    ax2.tick_params(colors='#CCCCCC')
    ax2.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    ax2.grid(True, axis='x', alpha=0.15, color='white')

    # Monthly charge & tenure info strip
    fig.text(0.5, 0.01,
             f'Monthly Charge: ${monthly_charge:.0f}  |  '
             f'Tenure: {int(tenure)} months  |  '
             f'Satisfaction: {int(satisfaction)}/5',
             ha='center', fontsize=9, color='#666688')

except FileNotFoundError as e:
    for ax in axes:
        ax.axis('off')
        ax.text(0.5, 0.5,
                f'PKL file not found.\nUpdate PKL_FOLDER path:\n{PKL_FOLDER}\n\n{e}',
                ha='center', va='center', transform=ax.transAxes,
                color='#e74c3c', fontsize=10)
except Exception as e:
    for ax in axes:
        ax.axis('off')
        ax.text(0.5, 0.5, f'Error:\n{e}',
                ha='center', va='center', transform=ax.transAxes,
                color='#e74c3c', fontsize=10)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
