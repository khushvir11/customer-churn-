# ============================================================
# POWER BI PYTHON VISUAL — Script 7
# Visual: Satisfaction Score × Category Heatmap +
#         Avg Monthly Charge per Category (side panel)
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer — analyses all rows.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pickle, re, warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix
from collections import defaultdict

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

fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                         gridspec_kw={'width_ratios':[2,1]})
fig.patch.set_facecolor(BG)

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else [False]*len(df)
    sub = df[has_reason].copy()

    sat_cat   = defaultdict(lambda: defaultdict(int))
    cat_charge= defaultdict(list)
    cat_tenure= defaultdict(list)

    for _, row in sub.iterrows():
        reason = str(row.get('Churn Reason',''))
        charge = float(row.get('Monthly Charge', 70))
        tenure = float(row.get('Tenure in Months', 12))
        sat    = int(row.get('Satisfaction Score', 2))
        clean  = clean_text(reason)
        tf     = tfidf.transform([clean])
        c,ps,ns,nu,po,su = sentiment(reason)
        meta   = np.array([[len(reason.split()),len(reason),c,ns,ps,nu,po,su,charge,tenure,sat]])
        probs  = model.predict_proba(hstack([tf,csr_matrix(meta)]).toarray())[0]
        top    = le.classes_[int(np.argmax(probs))]
        sat_cat[sat][top] += 1
        cat_charge[top].append(charge)
        cat_tenure[top].append(tenure)

    all_cats = list(CAT_COLORS.keys())
    sat_scores= sorted(sat_cat.keys())

    # Build heatmap matrix: rows=categories, cols=satisfaction scores
    matrix = np.zeros((len(all_cats), len(sat_scores)))
    for j, sat in enumerate(sat_scores):
        for i, cat in enumerate(all_cats):
            matrix[i, j] = sat_cat[sat].get(cat, 0)

    # ── LEFT: Heatmap ─────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#111128')
    im = ax1.imshow(matrix, aspect='auto', cmap='YlOrRd',
                    interpolation='nearest')
    ax1.set_xticks(range(len(sat_scores)))
    ax1.set_xticklabels([f'Score {s}' for s in sat_scores],
                         fontsize=9, color='#CCCCCC')
    ax1.set_yticks(range(len(all_cats)))
    ax1.set_yticklabels(all_cats, fontsize=10, color='#CCCCCC')

    for i in range(len(all_cats)):
        for j in range(len(sat_scores)):
            val = int(matrix[i,j])
            if val > 0:
                text_col = 'black' if matrix[i,j] > matrix.max()*0.6 else 'white'
                ax1.text(j, i, str(val), ha='center', va='center',
                         fontsize=11, fontweight='bold', color=text_col)

    ax1.set_title('Customer Count: Satisfaction Score × Churn Category',
                  fontsize=11, fontweight='bold', color='white', pad=10)
    ax1.set_xlabel('Satisfaction Score (1=Low → 5=High)',
                   fontsize=9, color='#AAAAAA')
    cbar = plt.colorbar(im, ax=ax1, shrink=0.75, pad=0.02)
    cbar.ax.tick_params(colors='#CCCCCC', labelsize=8)
    cbar.set_label('Customer Count', color='#AAAAAA', fontsize=8)

    # Grid lines between cells
    for x in np.arange(-0.5, len(sat_scores), 1):
        ax1.axvline(x, color='#1C1C2E', lw=1.5)
    for y in np.arange(-0.5, len(all_cats), 1):
        ax1.axhline(y, color='#1C1C2E', lw=1.5)

    # ── RIGHT: Avg charge + tenure per category ───────────────
    ax2 = axes[1]
    ax2.set_facecolor('#111128')

    avg_charges = [np.mean(cat_charge[c]) if cat_charge[c] else 0
                   for c in all_cats]
    avg_tenures = [np.mean(cat_tenure[c]) if cat_tenure[c] else 0
                   for c in all_cats]
    bar_colors  = [CAT_COLORS[c] for c in all_cats]

    x = np.arange(len(all_cats))
    w = 0.4
    b1 = ax2.bar(x - w/2, avg_charges, width=w,
                 color=bar_colors, edgecolor='#111128', alpha=0.9,
                 label='Avg Monthly Charge ($)')
    ax2_r = ax2.twinx()
    ax2_r.set_facecolor('#111128')
    b2 = ax2_r.bar(x + w/2, avg_tenures, width=w,
                   color=bar_colors, edgecolor='#111128', alpha=0.45,
                   label='Avg Tenure (months)', hatch='//')

    for bar, val in zip(b1, avg_charges):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+1,
                 f'${val:.0f}', ha='center', va='bottom',
                 fontsize=8, color='white')
    for bar, val in zip(b2, avg_tenures):
        ax2_r.text(bar.get_x()+bar.get_width()/2,
                   bar.get_height()+0.3,
                   f'{val:.0f}mo', ha='center', va='bottom',
                   fontsize=8, color='#AAAAAA')

    ax2.set_xticks(x)
    ax2.set_xticklabels(all_cats, rotation=20, ha='right',
                         fontsize=8, color='#CCCCCC')
    ax2.set_ylabel('Avg Monthly Charge ($)', fontsize=9, color='#AAAAAA')
    ax2_r.set_ylabel('Avg Tenure (months)', fontsize=9, color='#888899')
    ax2.set_title('Avg Charge & Tenure\nper Category',
                  fontsize=11, fontweight='bold', color='white', pad=10)
    ax2.tick_params(colors='#CCCCCC', labelsize=8)
    ax2_r.tick_params(colors='#888899', labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2_r.spines['top'].set_visible(False)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#333355')
    ax2.grid(True, axis='y', alpha=0.1, color='white', linestyle='--')

    lines1 = [plt.Line2D([0],[0], color=c, lw=8, alpha=0.9)
              for c in bar_colors]
    ax2.legend(lines1, all_cats, loc='upper right',
               frameon=False, labelcolor='#CCCCCC', fontsize=7)

except Exception as e:
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout()
plt.show()
