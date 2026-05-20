# ============================================================
# POWER BI PYTHON VISUAL — Script 2
# Visual: All 5 Category Probabilities (single customer)
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   Add Customer ID slicer (single select)
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    t = pos+neg+1e-9
    c = (pos-neg)/t
    return c,pos/t,neg/t,max(0,1-(pos+neg)/(len(toks)+1e-9)),c,min(1.0,(pos+neg)/(len(toks)+1e-9))

df = dataset.copy()

reason  = str(df['Churn Reason'].iloc[0])       if 'Churn Reason'       in df.columns else 'No reason'
charge  = float(df['Monthly Charge'].iloc[0])   if 'Monthly Charge'     in df.columns else 70.0
tenure  = float(df['Tenure in Months'].iloc[0]) if 'Tenure in Months'   in df.columns else 12.0
sat     = float(df['Satisfaction Score'].iloc[0])if 'Satisfaction Score' in df.columns else 2.0

fig, ax = plt.subplots(figsize=(7, 4.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    tf  = tfidf.transform([clean_text(reason)])
    c,ps,ns,nu,po,su = sentiment(reason)
    meta = np.array([[len(reason.split()),len(reason),c,ns,ps,nu,po,su,charge,tenure,sat]])
    probs = model.predict_proba(hstack([tf,csr_matrix(meta)]).toarray())[0]

    cats   = list(le.classes_)
    colors = [CAT_COLORS.get(c,'#aaaaaa') for c in cats]
    order  = np.argsort(probs)               # low → high for horizontal bar
    sorted_cats   = [cats[i]   for i in order]
    sorted_probs  = [probs[i]  for i in order]
    sorted_colors = [colors[i] for i in order]

    bars = ax.barh(sorted_cats, sorted_probs,
                   color=sorted_colors, edgecolor=BG,
                   height=0.55)

    # Highlight the top bar with a white border
    top_i = int(np.argmax(probs))
    top_local = list(order).index(top_i)
    bars[top_local].set_linewidth(2)
    bars[top_local].set_edgecolor('white')

    for bar, prob, cat in zip(bars, sorted_probs, sorted_cats):
        ax.text(bar.get_width() + 0.005,
                bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%',
                va='center', fontsize=11, color='white', fontweight='bold')

    ax.set_xlim(0, max(sorted_probs)*1.35)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlabel('Probability', fontsize=10, color='#AAAAAA')
    ax.tick_params(colors='#CCCCCC', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for sp in ['left','bottom']:
        ax.spines[sp].set_edgecolor('#333355')
    ax.grid(True, axis='x', alpha=0.12, color='white', linestyle='--')

    top_cat  = cats[top_i]
    top_prob = probs[top_i]
    ax.set_title(f'Churn Category Probabilities\nPredicted: {top_cat}  ({top_prob*100:.1f}%)',
                 fontsize=12, fontweight='bold', color='white', pad=10)

    reason_s = (reason[:65]+'…') if len(reason)>65 else reason
    fig.text(0.5, 0.01, f'"{reason_s}"', ha='center',
             fontsize=8, color='#666688', style='italic')

except Exception as e:
    ax.axis('off')
    ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout(rect=[0,0.05,1,1])
plt.show()
