# ============================================================
# POWER BI PYTHON VISUAL — SCRIPT 2
# Visual: Batch NLP Category Distribution (all churned customers)
#
# FILES NEEDED:
#   nlp_label_encoder.pkl
#   nlp_tfidf_vectorizer.pkl
#   nlp_catboost_model.pkl
#
# POWER BI SETUP:
#   Dataset: Telco_Full_PowerBI_NLP.csv
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer needed — shows all rows at once.
#   This visual works as a "category breakdown" page.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

CAT_COLORS = {
    'Competitor'     : '#e74c3c',
    'Price'          : '#e67e22',
    'Dissatisfaction': '#8e44ad',
    'Attitude'       : '#2980b9',
    'Other'          : '#27ae60',
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
    total = pos + neg + 1e-9
    compound = (pos - neg) / total
    return (compound, pos/total, neg/total,
            max(0, 1-(pos+neg)/(len(tokens)+1e-9)),
            compound, min(1.0, (pos+neg)/(len(tokens)+1e-9)))

def predict_batch(texts, charges, tenures, sats, tfidf, model, le):
    results = []
    for text, charge, tenure, sat in zip(texts, charges, tenures, sats):
        clean = clean_text(str(text))
        tf_vec = tfidf.transform([clean])
        compound, pos_s, neg_s, neu_s, pol, subj = simple_sentiment(str(text))
        meta = np.array([[
            len(str(text).split()), len(str(text)),
            compound, neg_s, pos_s, neu_s, pol, subj,
            float(charge), float(tenure), float(sat)
        ]])
        combined = hstack([tf_vec, csr_matrix(meta)]).toarray()
        probs    = model.predict_proba(combined)[0]
        top_idx  = np.argmax(probs)
        results.append({
            'category'  : le.classes_[top_idx],
            'confidence': probs[top_idx],
            'sentiment' : compound,
            'charge'    : float(charge),
            'probs'     : probs,
        })
    return results

# ── Main ─────────────────────────────────────────────────────
import pandas as pd
df = dataset.copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1C1C2E')
fig.suptitle('NLP Churn Category Analysis — All Churned Customers',
             fontsize=15, fontweight='bold', color='white', y=1.01)

try:
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
    with open(TF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    with open(CB_PATH, 'rb') as f:
        model = pickle.load(f)

    # Filter to rows with churn reason
    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else pd.Series([False]*len(df))
    churned_df = df[has_reason].copy()

    if len(churned_df) == 0:
        raise ValueError("No rows with 'Churn Reason' found in dataset.")

    texts   = churned_df['Churn Reason'].tolist()
    charges = churned_df['Monthly Charge'].fillna(70).tolist() \
        if 'Monthly Charge' in churned_df.columns else [70]*len(churned_df)
    tenures = churned_df['Tenure in Months'].fillna(12).tolist() \
        if 'Tenure in Months' in churned_df.columns else [12]*len(churned_df)
    sats    = churned_df['Satisfaction Score'].fillna(2).tolist() \
        if 'Satisfaction Score' in churned_df.columns else [2]*len(churned_df)

    results = predict_batch(texts, charges, tenures, sats, tfidf, model, le)

    categories   = [r['category']   for r in results]
    confidences  = [r['confidence'] for r in results]
    sentiments   = [r['sentiment']  for r in results]
    charge_vals  = [r['charge']     for r in results]

    cat_series = pd.Series(categories)
    cat_counts = cat_series.value_counts()
    colors     = [CAT_COLORS.get(c, '#aaaaaa') for c in cat_counts.index]

    # ── Plot 1: Category count bar ────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_facecolor('#252540')
    bars = ax1.barh(cat_counts.index, cat_counts.values,
                    color=colors, edgecolor='#1C1C2E', height=0.6)
    for bar, val in zip(bars, cat_counts.values):
        ax1.text(bar.get_width() + 1,
                 bar.get_y() + bar.get_height() / 2,
                 str(val), va='center', fontsize=11,
                 color='white', fontweight='bold')
    ax1.set_title('Churn Category Frequency', fontsize=12,
                  fontweight='bold', color='white')
    ax1.set_xlabel('Number of Customers', fontsize=10, color='#AAAAAA')
    ax1.tick_params(colors='#CCCCCC')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#333355')
    ax1.spines['bottom'].set_color('#333355')
    ax1.grid(True, axis='x', alpha=0.15, color='white')

    # ── Plot 2: Avg sentiment per category ────────────────────
    ax2 = axes[0, 1]
    ax2.set_facecolor('#252540')
    sent_df = pd.DataFrame({'category': categories, 'sentiment': sentiments})
    avg_sent = sent_df.groupby('category')['sentiment'].mean().sort_values()
    s_colors = ['#e74c3c' if v < -0.05 else
                '#f39c12' if v < 0.05 else '#27ae60'
                for v in avg_sent.values]
    ax2.barh(avg_sent.index, avg_sent.values,
             color=s_colors, edgecolor='#1C1C2E', height=0.55)
    ax2.axvline(0, color='#AAAAAA', lw=1.5, linestyle='--')
    for i, (cat, val) in enumerate(avg_sent.items()):
        ax2.text(val + (0.005 if val >= 0 else -0.005),
                 i, f'{val:+.2f}',
                 va='center', ha='left' if val >= 0 else 'right',
                 fontsize=10, color='white')
    ax2.set_title('Avg Sentiment by Category', fontsize=12,
                  fontweight='bold', color='white')
    ax2.set_xlabel('Avg Sentiment Score (-1=Negative, +1=Positive)',
                   fontsize=9, color='#AAAAAA')
    ax2.tick_params(colors='#CCCCCC')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#333355')
    ax2.spines['bottom'].set_color('#333355')
    ax2.grid(True, axis='x', alpha=0.15, color='white')

    # ── Plot 3: Revenue at risk per category ─────────────────
    ax3 = axes[1, 0]
    ax3.set_facecolor('#252540')
    rev_df = pd.DataFrame({'category': categories, 'charge': charge_vals})
    rev_risk = rev_df.groupby('category')['charge'].sum().sort_values()
    r_colors = [CAT_COLORS.get(c, '#aaaaaa') for c in rev_risk.index]
    bars3 = ax3.barh(rev_risk.index, rev_risk.values,
                     color=r_colors, edgecolor='#1C1C2E', height=0.55)
    for bar, val in zip(bars3, rev_risk.values):
        ax3.text(bar.get_width() + 5,
                 bar.get_y() + bar.get_height() / 2,
                 f'${val:,.0f}', va='center', fontsize=9, color='white')
    ax3.set_title('Monthly Revenue at Risk by Category', fontsize=12,
                  fontweight='bold', color='white')
    ax3.set_xlabel('Total Monthly Revenue ($)', fontsize=10, color='#AAAAAA')
    ax3.tick_params(colors='#CCCCCC')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_color('#333355')
    ax3.spines['bottom'].set_color('#333355')
    ax3.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.grid(True, axis='x', alpha=0.15, color='white')

    # ── Plot 4: Model confidence distribution ─────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#252540')
    conf_arr = np.array(confidences)
    ax4.hist(conf_arr, bins=20, color='#534AB7', edgecolor='#1C1C2E',
             alpha=0.85)
    ax4.axvline(conf_arr.mean(), color='#f39c12', linestyle='--',
                linewidth=2, label=f'Mean = {conf_arr.mean():.1%}')
    ax4.set_title('Model Confidence Distribution', fontsize=12,
                  fontweight='bold', color='white')
    ax4.set_xlabel('Prediction Confidence', fontsize=10, color='#AAAAAA')
    ax4.set_ylabel('Customer Count', fontsize=10, color='#AAAAAA')
    ax4.legend(fontsize=10, labelcolor='white',
               facecolor='#252540', edgecolor='#333355')
    ax4.tick_params(colors='#CCCCCC')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_color('#333355')
    ax4.spines['bottom'].set_color('#333355')
    ax4.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax4.grid(True, axis='y', alpha=0.15, color='white')

    total_rev = sum(charge_vals)
    fig.text(0.5, 0.0,
             f'Total customers analysed: {len(results):,}  |  '
             f'Total monthly revenue at risk: ${total_rev:,.2f}',
             ha='center', fontsize=10, color='#888899')

except FileNotFoundError as e:
    for ax in axes.flatten():
        ax.set_facecolor('#252540')
        ax.axis('off')
        ax.text(0.5, 0.5,
                f'PKL file not found.\nUpdate PKL_FOLDER:\n{PKL_FOLDER}\n\n{e}',
                ha='center', va='center', transform=ax.transAxes,
                color='#e74c3c', fontsize=10)
except Exception as e:
    for ax in axes.flatten():
        ax.set_facecolor('#252540')
        ax.axis('off')
        ax.text(0.5, 0.5, f'Error:\n{e}',
                ha='center', va='center', transform=ax.transAxes,
                color='#e74c3c', fontsize=10)

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.show()
