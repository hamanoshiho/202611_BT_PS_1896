import pandas as pd, re, numpy as np
U='/mnt/user-data/uploads/Downloads/'
def clean(text):
    text = re.sub(r"={3,}\s*PDF-Seite\s+\d+\s*={3,}", " ", text)
    text = re.sub(r"(?m)^\s*\d+\s+Psychische Studien\..*$", " ", text)
    text = text.replace("ſ", "s")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()
N=80
rows=[]
a=pd.read_csv(U+'psychische_studien_v23_1896_articles.csv',encoding='utf-8-sig')
a=a[a.article_type!='advertisement']
for _,r in a.iterrows():
    w=clean(str(r.text)).split()
    for i in range(0,len(w),N):
        seg=w[i:i+N]
        if len(seg)<30: continue
        rows.append(dict(pid=f'{r.id}_p{i//N:03d}',medium='PS',src=r.id,date=r.issue,month=r.issue,title=r.title,article_type=r.article_type,bt_topic=None,cats='',text=' '.join(seg)))
bt=pd.read_pickle('bt_sel.pkl'); bt=bt[bt['any']]
P={'Röntgen・X線':r'r[öo][nu]tgen|iöntgen|x[- ]?[sf]trahl|kathoden[sf]trahl',
'心霊主義':r'sp[il1]r[il1][tl][il1][sfß]m|sp[il1]r[il1][tl][il1][sfß]t|spiritual[il1][sf]|s[ée]ance|materiali[sf]ation|tisch(rück|rnck)|klopfgei[sf]t|geisterklopf',
'霊媒':r'\bmedi[un][mn]\b',
'オカルト・神智学':r'[o0][kc][kc]ult|theo[sf][o0]ph',
'催眠・動物磁気':r'h[yv]pn[o0]t|hhpno|s[o0]mnamb|magneti[sf]eur|me[sf]meri|heilmagnet|[lt]ebensmagnet|thieri[sf]che[nm]? magneti',
'透視・テレパシー':r'hell[sf]eh[ec]r|telepath|gedankenle[sf][ec]r|gedankenübertr|wahrtraum',
'幽霊・心霊現象':r'spukhau[sf]|spukt\b|spuker[sf]ch|spukge[sf]ch|gei[sf]terseh|gei[sf]tererschein|gei[sf]terhau[sf]|gei[sf]terphot|gei[sf]terbe[sf]chw|gespen[sf]terersch',
'Reichenbach・Od':r'\bodi[sf]ch|reichenbach|\bodlicht',
'占星・魔女・黒魔術':r'a[sf]trolog|hexenproze|hexenwahn|[sf]chwarzkün[sf]t'}
cats=[c for c in P if c in bt.columns]
for _,r in bt.iterrows():
    w=re.sub(r'\s+',' ',r.text).split(); lw=[x.lower() for x in w]
    hits=[]
    for c in cats:
        if not r[c]: continue
        for i,x in enumerate(lw):
            if re.search(P[c],x) or (c=='Reichenbach・Od' and re.search(P[c],x)): hits.append((i,c))
    if not hits:  # multiword patterns
        hits=[(len(w)//2,[c for c in cats if r[c]][0])]
    hits.sort(); wins=[]
    for i,c in hits:
        s=max(0,i-N//2); e=min(len(w),s+N); s=max(0,e-N)
        if wins and s<wins[-1][1]: wins[-1][1]=max(wins[-1][1],e); wins[-1][2].add(c)
        else: wins.append([s,e,{c}])
    for j,(s,e,cs) in enumerate(wins):
        # split merged windows into N chunks
        for k in range(s,e,N):
            seg=w[k:k+N]
            if len(seg)<30 and k>s: continue
            rows.append(dict(pid=f'{r.id}_w{j}{(k-s)//N}',medium='BT',src=r.id,date=r.date,month=r.month,title='',article_type='',bt_topic=int(r.topic),cats='; '.join(sorted(cs)),text=' '.join(seg)))
D=pd.DataFrame(rows); print(D.medium.value_counts()); print(D.groupby('medium').text.apply(lambda s:s.str.split().str.len().mean()))
D.to_csv('/mnt/user-data/outputs/joint_passages.csv',index=False,encoding='utf-8-sig')
import pandas as pd, numpy as np, warnings, pickle, sys; warnings.filterwarnings('ignore')
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
import umap, hdbscan
from stopwordsiso import stopwords
U='/mnt/user-data/uploads/Downloads/media_compare_1896/'
D=pd.read_csv(U+'joint_passages.csv',encoding='utf-8-sig'); E=np.load(U+'joint_passages_embeddings.npy')
extra=set("""abend abg aln ann april auch august aun aus bat berlin berliner berlinn ctr dan das der dezember die din dis dr durck ent februar frau frl für geb gestern habe herr herrn heute hrn hru hätte icht inen jahre jahres januar juli juni mac mai med mehr morgen märz nachm november nr och off oktober orn per präs sei seien seite september sick str tageblatt ten the theil theile uhr unt ver vorm wnrde worden wurde wurden zeit psychische studien psychischen jahrg jahrgang heft seiten redaction redaktion uud nnd eiu vou anf sür ans uicht nnr dcr dcs eiue vird sic bci uud iu cr""".split())
sw=list(set(stopwords('de'))|extra)
MCS=int(sys.argv[1]) if len(sys.argv)>1 else 15
def run(seed):
    vec=CountVectorizer(stop_words=sw,ngram_range=(1,2),min_df=3,max_df=0.5,token_pattern=r"(?u)\b[^\W\d_]{3,}\b")
    m=BERTopic(umap_model=umap.UMAP(n_neighbors=15,n_components=5,min_dist=0.0,metric='cosine',random_state=seed),
      hdbscan_model=hdbscan.HDBSCAN(min_cluster_size=MCS,min_samples=5,metric='euclidean',prediction_data=True),
      vectorizer_model=vec,ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),embedding_model=None,language='multilingual')
    t,_=m.fit_transform(D.text.tolist(),E); return m,np.array(t)
runs=[run(s) for s in (42,7,123)]
m,t=runs[0]
print('MCS',MCS,'topics',len(set(t))-1,'outliers',(t==-1).sum(),'ARI',[round(adjusted_rand_score(t,r[1]),2) for r in runs[1:]])
ct=pd.crosstab(t,D.medium)
for k in sorted(set(t)):
    if k==-1: print(-1,dict(ct.loc[k])); continue
    print(k,dict(ct.loc[k]),', '.join(w for w,_ in m.get_topic(k)[:10]))
pickle.dump({'t':t,'runs':[r[1] for r in runs]},open(f'j2_mcs{MCS}.pkl','wb')); m.save(f'joint2_mcs{MCS}',serialization='pickle',save_ctfidf=True)
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
from bertopic import BERTopic
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Noto Sans CJK JP'
U='/mnt/user-data/uploads/Downloads/media_compare_1896/'
D=pd.read_csv(U+'joint_passages.csv',encoding='utf-8-sig')
R=pickle.load(open('j2_mcs25.pkl','rb')); t=R['t']; D['joint_topic']=t
m=BERTopic.load('joint2_mcs25')
L={0:'Düsseldorf心霊裁判',1:'唯物論批判・世界観',2:'書籍・文献案内',3:'叩音・音楽的現象（Müller／Femme masquée）',4:'Eusapia Paladino事件',5:'夜の怪異体験（室内）',6:'Röntgen写真',7:'Johannes Faust（魔術史）',8:'放電管・光線・電気（物理）',9:'民間伝承・自然の怪異（森・山）',10:'Boy Spuk・死の予兆',11:'ロシアの心霊主義（Aksakow）',12:'鬼火・夜の怪光',13:'オカルティスト連盟と大会',14:'心霊主義運動の普及',15:'真理・認識をめぐる論争',16:'Jarischauの夜の怪光（Wittig連載）',17:'国際心理学会議',18:'催眠・夢遊（医学）',19:'教会・聖職者',20:'魔術・魔法の歴史',21:'Reichelの磁気治療',22:'物質・エネルギー・運動',23:'幽霊目撃（家・子供）',24:'占星術・流体・Od',25:'Faust本・文献史',26:'動物磁気と診断',27:'Swedenborg・宗教思想'}
# stability: best Jaccard with other seeds
def bestj(a,b,k):
    A=set(np.where(a==k)[0]); best=0
    for j in set(b)-{-1}:
        B=set(np.where(b==j)[0]); best=max(best,len(A&B)/len(A|B))
    return best
base=len(D[D.medium=='BT'])/len(D)
rows=[]
for k in sorted(set(t)-{-1}):
    s=D[D.joint_topic==k]; nb=(s.medium=='BT').sum(); npx=(s.medium=='PS').sum()
    share=nb/len(s)
    typ='共通' if nb>=5 and npx>=5 else ('PS中心' if share<0.5 else 'BT中心')
    rows.append(dict(topic=k,ラベル=L[k],上位語=', '.join(w for w,_ in m.get_topic(k)[:10]),BT=nb,PS=npx,BT比率=round(share,3),
      BT比率_対期待値=round(share/base,2),区分=typ,安定性_Jaccard=round(np.mean([bestj(t,r,k) for r in R['runs'][1:]]),2),
      BT月=' '.join(f'{a[5:]}:{b}' for a,b in s[s.medium=='BT'].month.value_counts().sort_index().items()),
      PS号=' '.join(f'{a[5:]}:{b}' for a,b in s[s.medium=='PS'].month.value_counts().sort_index().items())))
T=pd.DataFrame(rows).sort_values(['区分','BT'],ascending=[True,False])
pd.set_option('display.width',300); pd.set_option('display.max_colwidth',60)
print(T.drop(columns=['上位語']).to_string())
print('base',round(base,3),'outliers BT',((D.medium=='BT')&(t==-1)).sum(),'/',(D.medium=='BT').sum(),' PS',((D.medium=='PS')&(t==-1)).sum(),'/',(D.medium=='PS').sum())
D['joint_label']=D.joint_topic.map(L).fillna('外れ値')
pickle.dump((D,T,L),open('j3.pkl','wb'))
import pandas as pd, numpy as np, pickle, warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family']='Noto Sans CJK JP'
D,T,L=pickle.load(open('j3.pkl','rb'))
E=np.load('/mnt/user-data/uploads/Downloads/media_compare_1896/joint_passages_embeddings.npy')
base=(D.medium=='BT').mean()
def tier(r):
    if r.BT>=5 and r.BT比率_対期待値>=1: return '① 共通の主題'
    if r.BT>0: return '② PS中心（BTは散発）'
    return '③ PSのみ'
T['区分']=T.apply(tier,axis=1)
note={2:'内容ではなく記事の形式による共通。PSは「Bibliographie-Nachweis」と「Kurze Notizen」、BTは雑報や催しの告知',6:'種を変えると8と一つのトピックにまとまる（6+8の組み合わせは3つの種すべてで安定）',8:'同上',21:'BT側の5件はRöntgen線の医療利用・戦傷の記事で、Reichelではない。共通とみなすのは保留',0:'BT側の7件は議会・政治記事で、裁判とは無関係（OCRの段組み混在による誤った割り当て）',13:'BT側の5件はいずれも2月の霊媒関連記事で、大会の記事ではない'}
T['確認メモ']=T.topic.map(note).fillna('')
T.loc[T.topic==2,'区分']='①b 形式上の共通（雑報・文献案内）'
L[2]='雑報・文献案内'; T.loc[T.topic==2,'ラベル']=L[2]
T.loc[T.topic==21,'区分']='② PS中心（BTは散発）'
T=T.sort_values(['区分','BT','PS'],ascending=[True,False,False])
# representative passages
reps=[]
for k in T[T.区分.str.startswith('①')].topic:
    idx=np.where(D.joint_topic.values==k)[0]; c=E[idx].mean(0); c/=np.linalg.norm(c)
    for md in ['BT','PS']:
        ii=[i for i in idx if D.medium.iat[i]==md]
        for i in sorted(ii,key=lambda i:-E[i]@c)[:4]:
            r=D.iloc[i]; reps.append(dict(topic=k,ラベル=L[k],媒体=md,日付・号=r.date,PS記事題=r.title if md=='PS' else '',BT検出カテゴリ=r.cats if md=='BT' else '',重心との類似度=round(float(E[i]@c),3),本文=r.text))
REP=pd.DataFrame(reps)
# monthly for shared
sh=[6,8,18,2]
mon=[f'1896-{m:02d}' for m in range(1,13)]
MT=[]
for k in sh:
    for md in ['BT','PS']:
        s=D[(D.joint_topic==k)&(D.medium==md)].month.value_counts()
        MT.append({'topic':k,'ラベル':L[k],'媒体':md,**{m[5:]:int(s.get(m,0)) for m in mon}})
MT=pd.DataFrame(MT)
# combined Röntgen 6+8
summ=pd.DataFrame({'項目':['目的','コーパス','分析単位','埋め込み','モデル','安定性','外れ値','主な限界'],'内容':[
 '一般向け日刊紙（BT）と心霊研究誌（PS）の1896年の記事を一つのトピックモデルに入れ、両誌が同じトピックを共有しているかを直接確かめる',
 'BT: Röntgen線・心霊主義・催眠・透視・オカルト・幽霊・Od・占星などの語を含む188文書（OCR誤りを考慮した正規表現で抽出。競馬の馬名「Spiritist」などは除外）／PS: 第23巻の全記事（広告を除く93記事）',
 '約80語の文章片（パッセージ）。BTは検出語を中心にした前後40語ずつの窓、PSは記事を80語ごとに分割 → BT 283件、PS 3,087件。80語にしたのは、埋め込みモデルが読める先頭128トークンに収め、文章片全体が埋め込みに反映されるようにするため',
 'paraphrase-multilingual-MiniLM-L12-v2（これまでの2つの分析と同じモデル。Macのキャッシュから実行）',
 'BERTopic：UMAP(n_neighbors=15, n_components=5, min_dist=0, cosine, random_state=42)、HDBSCAN(min_cluster_size=25, min_samples=5)、CountVectorizer(1–2gram, min_df=3, max_df=0.5, ドイツ語ストップワード＋BTで追加したストップワード)。min_cluster_size=15でも試し、結果はほぼ同じ',
 '種42と、種7・123の結果との一致度（ARI）は0.63・0.67。各トピックの安定性は、他の種で最もよく重なるトピックとのJaccard係数で示した（トピック一覧の列）',
 f'外れ値（-1）はBT 119/283件（42%）、PS 1,016/3,087件（33%）',
 'BTはFraktur体のOCRで段組みが混ざるため、窓の中に無関係な記事の語が入ることがある。BTの文章片が少数しかないトピックでは誤った割り当てがあるので確認メモを参照。トピックのラベルは上位語と代表的な文章片から付けた仮のもの']})
with pd.ExcelWriter('out2/BT_PS_1896_joint_model.xlsx') as w:
    summ.to_excel(w,sheet_name='概要',index=False)
    T.to_excel(w,sheet_name='トピック一覧',index=False)
    REP.to_excel(w,sheet_name='共通トピック代表文',index=False)
    MT.to_excel(w,sheet_name='共通トピック月別',index=False)
    D[['pid','medium','src','date','title','cats','bt_topic','joint_topic','joint_label','text']].to_excel(w,sheet_name='全パッセージ割当',index=False)
# fig1
T2=T.copy(); T2['name']=T2.apply(lambda r:f'{r.topic} {r.ラベル}',axis=1)
T2=T2.iloc[::-1]
fig,ax=plt.subplots(figsize=(10,9))
y=np.arange(len(T2))
ax.barh(y,T2.PS,color='#9aa5b1',label='Psychische Studien')
ax.barh(y,T2.BT,left=T2.PS,color='#c1121f',label='Berliner Tageblatt')
for i,(p,b,r) in enumerate(zip(T2.PS,T2.BT,T2.BT比率)):
    if b: ax.text(p+b+3,i,f'BT {b}（{r*100:.0f}%）',va='center',fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(T2.name,fontsize=8.5)
cols={'①':'#c1121f','②':'#333','③':'#888'}
for lbl,tr in zip(ax.get_yticklabels(),T2.区分): lbl.set_color(cols[tr[0]]); 
ax.set_xlabel('文章片の数'); ax.legend(loc='lower right')
ax.set_title(f'合同モデルの各トピックにおける媒体の内訳（BTの全体比 {base*100:.1f}%）\n赤字 = 共通（BTが5件以上、かつ全体比以上。2は形式による共通）')
fig.tight_layout(); fig.savefig('out2/fig_joint_topic_composition.png',dpi=160)
# fig2
fig,axs=plt.subplots(2,2,figsize=(12,6.5),sharex=True)
for ax,k in zip(axs.flat,sh):
    for md,col in [('BT','#c1121f'),('PS','#5c6b7a')]:
        r=MT[(MT.topic==k)&(MT.媒体==md)].iloc[0]
        ax.plot([m[5:] for m in mon],[r[m[5:]] for m in mon],marker='o',color=col,label=md)
    ax.set_title(f'{k} {L[k]}',fontsize=10); ax.grid(alpha=.3)
axs[0,0].legend(); 
for ax in axs[1]: ax.set_xlabel('1896年 月（PSは号）')
fig.supylabel('文章片の数'); fig.tight_layout(); fig.savefig('out2/fig_shared_topics_by_month.png',dpi=160)
print(T[['topic','ラベル','BT','PS','BT比率_対期待値','区分','安定性_Jaccard']].to_string())
