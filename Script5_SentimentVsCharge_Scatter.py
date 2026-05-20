# ============================================================
# POWER BI PYTHON VISUAL — Script 5
# Visual: Sentiment Score vs Monthly Charge Scatter
#         (coloured by predicted churn category)
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer — plots all churned customers at once.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor('#111128')

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else [False]*len(df)
    sub = df[has_reason].copy()

    records = []
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
        records.append({'cat': top, 'sentiment': c, 'charge': charge, 'tenure': tenure})

    for cat, col in CAT_COLORS.items():
        pts = [r for r in records if r['cat']==cat]
        if not pts: continue
        xs = [p['sentiment'] for p in pts]
        ys = [p['charge']    for p in pts]
        ss = [max(20, p['tenure']*2) for p in pts]
        ax.scatter(xs, ys, c=col, s=ss, alpha=0.65,
                   edgecolors=col, linewidths=0.5, label=cat, zorder=3)

    # Reference lines
    ax.axvline(0, color='#555577', lw=1.2, linestyle='--', alpha=0.6)
    ax.axhline(np.mean([r['charge'] for r in records]),
               color='#555577', lw=1.2, linestyle='--', alpha=0.6)

    ax.set_xlabel('Sentiment Score  (← Negative  |  Positive →)',
                  fontsize=10, color='#AAAAAA')
    ax.set_ylabel('Monthly Charge ($)', fontsize=10, color='#AAAAAA')
    ax.set_title('Churn Reason Sentiment vs Monthly Charge\n(bubble size = tenure months)',
                 fontsize=12, fontweight='bold', color='white', pad=10)

    ax.tick_params(colors='#CCCCCC', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left','bottom']:
        ax.spines[sp].set_edgecolor('#333355')
    ax.grid(True, alpha=0.1, color='white', linestyle='--')

    legend_handles = [mpatches.Patch(color=CAT_COLORS[c], label=c)
                      for c in CAT_COLORS]
    ax.legend(handles=legend_handles, loc='upper right',
              frameon=False, labelcolor='#CCCCCC', fontsize=9)

    # Quadrant labels
    ax.text(0.02, 0.97, 'Negative\nsentiment', transform=ax.transAxes,
            fontsize=8, color='#e74c3c', va='top', alpha=0.7)
    ax.text(0.72, 0.97, 'Positive\nsentiment', transform=ax.transAxes,
            fontsize=8, color='#27ae60', va='top', alpha=0.7)

except Exception as e:
    ax.axis('off')
    ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout()
plt.show()
