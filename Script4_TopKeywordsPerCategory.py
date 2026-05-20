# ============================================================
# POWER BI PYTHON VISUAL — Script 4
# Visual: Top Keywords per Churn Category (from TF-IDF vocab)
#
# POWER BI SETUP:
#   Drag into "Values": Churn Reason, Monthly Charge,
#                       Tenure in Months, Satisfaction Score
#   No slicer needed — analyses all rows.
# ============================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

try:
    with open(LE_PATH,'rb') as f: le = pickle.load(f)
    with open(TF_PATH,'rb') as f: tfidf = pickle.load(f)
    with open(CB_PATH,'rb') as f: model = pickle.load(f)

    has_reason = df['Churn Reason'].notna() if 'Churn Reason' in df.columns else [False]*len(df)
    sub = df[has_reason].copy()

    vocab      = tfidf.get_feature_names_out()
    cat_tfidf  = defaultdict(lambda: np.zeros(len(vocab)))

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
        cat_tfidf[top] += tf.toarray()[0]

    all_cats = list(CAT_COLORS.keys())
    n_cats   = len(all_cats)
    fig, axes = plt.subplots(1, n_cats, figsize=(16, 4.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle('Top Keywords Driving Each Churn Category',
                 fontsize=13, fontweight='bold', color='white', y=1.02)

    TOP_N = 6
    for ax, cat in zip(axes, all_cats):
        ax.set_facecolor('#111128')
        scores = cat_tfidf[cat]
        if scores.sum() == 0:
            ax.axis('off')
            ax.text(0.5,0.5,'No data', ha='center', va='center',
                    transform=ax.transAxes, color='#888899')
            ax.set_title(cat, fontsize=10, fontweight='bold',
                         color=CAT_COLORS[cat], pad=8)
            continue

        top_idx   = np.argsort(scores)[-TOP_N:]
        top_words = [vocab[i] for i in top_idx]
        top_scores= [scores[i] for i in top_idx]
        col       = CAT_COLORS[cat]

        norm_scores = np.array(top_scores)
        norm_scores = norm_scores / (norm_scores.max() + 1e-9)
        bar_colors  = [col + hex(int(60 + 195*s))[2:].zfill(2) for s in norm_scores]

        bars = ax.barh(top_words, norm_scores,
                       color=bar_colors, edgecolor='#111128', height=0.6)
        for bar, sc in zip(bars, norm_scores):
            ax.text(bar.get_width()+0.02,
                    bar.get_y()+bar.get_height()/2,
                    f'{sc:.2f}', va='center', fontsize=8, color='#CCCCCC')

        ax.set_xlim(0, 1.35)
        ax.set_title(cat, fontsize=11, fontweight='bold',
                     color=col, pad=8)
        ax.tick_params(colors='#CCCCCC', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for sp in ['left','bottom']:
            ax.spines[sp].set_edgecolor('#333355')
        ax.set_xlabel('Relative TF-IDF', fontsize=8, color='#888899')
        ax.grid(True, axis='x', alpha=0.1, color='white', linestyle='--')

except Exception as e:
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis('off')
    ax.text(0.5,0.5,f'Error:\n{e}', ha='center', va='center',
            transform=ax.transAxes, color='#e74c3c', fontsize=10)

plt.tight_layout()
plt.show()
