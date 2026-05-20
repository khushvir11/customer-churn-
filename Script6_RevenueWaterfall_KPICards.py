# ============================================================
# POWER BI PYTHON VISUAL — Script 6
# Visual: Revenue at Risk Waterfall by Churn Category
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer — shows full revenue breakdown.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import pickle, re, warnings
warnings.filterwarnings('ignore')
from scipy.sparse import hstack, csr_matrix

PKL_FOLDER = r"C:\Users\khush\Downloads"
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

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else [False]*len(df)
    sub = df[has_reason].copy()

    cat_rev   = {c: 0.0 for c in CAT_COLORS}
    cat_count = {c: 0   for c in CAT_COLORS}

    for _, row in sub.iterrows():
        reason = str(row.get('Churn Reason',''))
        charge = float(row.get('Monthly Charge', 70))
        tenure = float(row.get('Tenure in Months', 12))
        sat    = float(row.get('Satisfaction Score', 2))
        clean  = clean_text(reason)
        tf     = tfidf.transform([clean])
        c,ps,ns,nu,po,su = sentiment(reason)
        meta   = np.array([[len(reason.split()),len(reason),c,ns,ps,nu,po,su,charge,tenure,sat]])
        probs  = model.predict_proba(hstack([tf,csr_matrix(meta)]).toarray())[0]
        top    = le.classes_[int(np.argmax(probs))]
        cat_rev[top]   += charge
        cat_count[top] += 1

    all_cats = list(CAT_COLORS.keys())
    revenues = [cat_rev[c] for c in all_cats]
    counts   = [cat_count[c] for c in all_cats]
    colors   = [CAT_COLORS[c] for c in all_cats]
    total    = sum(revenues)

    # ── LEFT: Waterfall chart ─────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#111128')

    # Sort by revenue descending for waterfall
    order = np.argsort(revenues)[::-1]
    s_cats = [all_cats[i] for i in order]
    s_revs = [revenues[i] for i in order]
    s_cols = [colors[i]   for i in order]

    running = 0
    wf_labels = s_cats + ['TOTAL']
    wf_bottoms= []
    wf_heights= []
    wf_colors = []

    for rev in s_revs:
        wf_bottoms.append(running)
        wf_heights.append(rev)
        running += rev
    # Total bar
    wf_bottoms.append(0)
    wf_heights.append(total)

    for i, (bot, h, cat) in enumerate(zip(wf_bottoms, wf_heights,
                                           s_cats + ['TOTAL'])):
        col = CAT_COLORS.get(cat, '#534AB7')
        bar = ax1.bar(i, h, bottom=bot, color=col,
                      edgecolor='#111128', width=0.6, alpha=0.9)
        # Connector line
        if i < len(s_cats):
            ax1.plot([i+0.3, i+0.7], [bot+h, bot+h],
                     color='#555577', lw=1, linestyle='--')
        # Value label
        ax1.text(i, bot+h+total*0.012,
                 f'${h:,.0f}', ha='center', va='bottom',
                 fontsize=8.5, color='white', fontweight='bold')

    ax1.set_xticks(range(len(wf_labels)))
    ax1.set_xticklabels(wf_labels, rotation=20, ha='right',
                         fontsize=9, color='#CCCCCC')
    ax1.set_ylabel('Monthly Revenue at Risk ($)', fontsize=9, color='#AAAAAA')
    ax1.set_title('Revenue at Risk — Waterfall by Category',
                  fontsize=11, fontweight='bold', color='white', pad=10)
    ax1.tick_params(colors='#CCCCCC', labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for sp in ['left','bottom']:
        ax1.spines[sp].set_edgecolor('#333355')
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_:f'${x:,.0f}'))
    ax1.grid(True, axis='y', alpha=0.1, color='white', linestyle='--')

    # ── RIGHT: KPI cards grid ─────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG); ax2.axis('off')

    ax2.set_title('Category KPI Summary', fontsize=11,
                  fontweight='bold', color='white', pad=10)

    card_data = sorted(
        [(c, cat_count[c], cat_rev[c]) for c in all_cats],
        key=lambda x: x[2], reverse=True
    )

    n = len(card_data)
    card_h = 0.85 / n
    for i, (cat, cnt, rev) in enumerate(card_data):
        y_top = 0.92 - i * card_h
        col   = CAT_COLORS[cat]

        # Card background
        rect = mpatches.FancyBboxPatch(
            (0.02, y_top - card_h + 0.01), 0.96, card_h - 0.02,
            boxstyle='round,pad=0.01',
            facecolor=col+'22', edgecolor=col+'66',
            linewidth=1, transform=ax2.transAxes, clip_on=False
        )
        ax2.add_patch(rect)

        # Category name
        ax2.text(0.08, y_top - card_h/2 + 0.025, cat,
                 transform=ax2.transAxes,
                 fontsize=10, fontweight='bold', color=col, va='center')
        # Customers
        ax2.text(0.50, y_top - card_h/2 + 0.025,
                 f'{cnt} customers',
                 transform=ax2.transAxes,
                 fontsize=9, color='#CCCCCC', va='center')
        # Revenue
        ax2.text(0.78, y_top - card_h/2 + 0.025,
                 f'${rev:,.0f}/mo',
                 transform=ax2.transAxes,
                 fontsize=9, fontweight='bold', color='white', va='center')
        # Share bar
        share = rev / (total + 1e-9)
        ax2.barh([y_top - card_h + 0.015], [share * 0.9],
                 left=0.06, height=0.012, color=col, alpha=0.5,
                 transform=ax2.transAxes)

    fig.text(0.75, 0.02,
             f'Total monthly revenue at risk: ${total:,.2f}',
             ha='center', fontsize=9, color='#888899')

except Exception as e:
    for ax in axes:
        ax.set_facecolor(BG); ax.axis('off')
        ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
                transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout()
plt.show()
