# ============================================================
# POWER BI PYTHON VISUAL — Script 3
# Visual: Churn Category Distribution Donut (all customers)
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer needed — shows full dataset breakdown.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle, re, warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix

PKL_FOLDER = r"C:\PowerBI_Telco"
LE_PATH = PKL_FOLDER + r"\nlp_label_encoder.pkl"
TF_PATH = PKL_FOLDER + r"\nlp_tfidf_vectorizer.pkl"
CB_PATH = PKL_FOLDER + r"\nlp_catboost_model.pkl"

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
    'Attitude':'#2980b9','Competitor':'#e74c3c',
    'Dissatisfaction':'#8e44ad','Other':'#27ae60','Price':'#e67e22',
}
BG = '#1C1C2E'

def clean_text(t):
    t = re.sub(r'[^a-z\s]',' ',str(t).lower())
    return ' '.join(w for w in re.sub(r'\s+',' ',t).strip().split()
                    if w not in STOP_WORDS and len(w)>2)

def sentiment(text):
    toks = text.lower().split()
    pos = sum(1 for w in toks if w in POS_WORDS)
    neg = sum(1 for w in toks if w in NEG_WORDS)
    t = pos+neg+1e-9; c=(pos-neg)/t
    return c,pos/t,neg/t,max(0,1-(pos+neg)/(len(toks)+1e-9)),c,min(1.0,(pos+neg)/(len(toks)+1e-9))

df = dataset.copy()

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor(BG)

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else [False]*len(df)
    sub = df[has_reason].copy()

    cats_pred, rev_risk = [], []
    for _, row in sub.iterrows():
        reason = str(row.get('Churn Reason',''))
        charge = float(row.get('Monthly Charge', 70))
        tenure = float(row.get('Tenure in Months', 12))
        sat    = float(row.get('Satisfaction Score', 2))
        tf = tfidf.transform([clean_text(reason)])
        c,ps,ns,nu,po,su = sentiment(reason)
        meta = np.array([[len(reason.split()),len(reason),c,ns,ps,nu,po,su,charge,tenure,sat]])
        probs = model.predict_proba(hstack([tf,csr_matrix(meta)]).toarray())[0]
        top   = le.classes_[int(np.argmax(probs))]
        cats_pred.append(top)
        rev_risk.append(charge)

    import pandas as pd
    res_df = pd.DataFrame({'category': cats_pred, 'revenue': rev_risk})
    counts = res_df['category'].value_counts()
    all_cats   = list(CAT_COLORS.keys())
    count_vals = [counts.get(c,0) for c in all_cats]
    colors     = [CAT_COLORS[c] for c in all_cats]
    total      = sum(count_vals)

    # ── LEFT: Donut ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(BG)
    wedges, texts, autotexts = ax1.pie(
        count_vals, labels=None,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=colors,
        startangle=90,
        wedgeprops={'width':0.55,'edgecolor':BG,'linewidth':2},
        pctdistance=0.78,
        textprops={'fontsize':9,'color':'white'}
    )
    for at in autotexts:
        at.set_fontweight('bold')
    centre = plt.Circle((0,0), 0.42, fc=BG)
    ax1.add_patch(centre)
    ax1.text(0, 0.08, str(total), ha='center', va='center',
             fontsize=22, fontweight='bold', color='white')
    ax1.text(0,-0.18, 'Churned', ha='center', va='center',
             fontsize=10, color='#888899')
    ax1.set_title('Category Distribution', fontsize=12,
                  fontweight='bold', color='white', pad=10)

    # Custom legend
    legend_items = [plt.Line2D([0],[0], marker='o', color='w',
                    markerfacecolor=CAT_COLORS[c], markersize=8,
                    label=f'{c}  ({counts.get(c,0):,})')
                    for c in all_cats if counts.get(c,0)>0]
    ax1.legend(handles=legend_items, loc='lower center',
               bbox_to_anchor=(0.5,-0.18), ncol=2,
               frameon=False, labelcolor='#CCCCCC', fontsize=9)

    # ── RIGHT: Revenue at risk per category ───────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    rev_by_cat = res_df.groupby('category')['revenue'].sum().reindex(all_cats, fill_value=0)
    bar_colors = [CAT_COLORS[c] for c in all_cats]
    bars = ax2.bar(all_cats, rev_by_cat.values,
                   color=bar_colors, edgecolor=BG, width=0.55)
    for bar, val in zip(bars, rev_by_cat.values):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+20,
                 f'${val:,.0f}', ha='center', va='bottom',
                 fontsize=9, color='white', fontweight='bold')
    ax2.set_ylabel('Monthly Revenue ($)', fontsize=10, color='#AAAAAA')
    ax2.set_title('Revenue at Risk by Category', fontsize=12,
                  fontweight='bold', color='white', pad=10)
    ax2.tick_params(colors='#CCCCCC', labelsize=9, axis='x', rotation=15)
    ax2.tick_params(colors='#CCCCCC', labelsize=9, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for sp in ['left','bottom']:
        ax2.spines[sp].set_edgecolor('#333355')
    ax2.grid(True, axis='y', alpha=0.12, color='white', linestyle='--')
    ax2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x,_: f'${x:,.0f}'))

except Exception as e:
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout()
plt.show()
