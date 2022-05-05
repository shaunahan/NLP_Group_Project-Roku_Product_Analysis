# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:07:18 2022

@author: zsk98
"""

def my_stop_words(var_in, new=[]):
    from nltk.corpus import stopwords
    sw = stopwords.words('english')
    for i in new:
        sw.append(i)
    tmp = [word for word in str(var_in).split() if word not in sw]
    tmp = ' '.join(tmp)
    return tmp

def clean_txt(txt_in,n):
    import re
    from langdetect import detect
    import pandas as pd
    
    t=[]
    for i in txt_in:
        #print(i)
        #print(type(i))
        clean_str = re.sub("[^A-Za-z]+", " ", str(i)).strip().lower()
        t.append(clean_str)
    
    txt_in1=[i for i in t if len(i.split())>n] #getting rid of comments that has less than n words

    txt_in2=[i for i in txt_in1 if detect(i)=="en"]
    
    ret=pd.DataFrame(txt_in2,columns=["bd_cleaned"])

    return ret

def my_stem(var_in):
    from nltk.stem.porter import PorterStemmer
    my_stem = PorterStemmer()
    tmp = [my_stem.stem(word) for word in var_in.split()]
    tmp = ' '.join(tmp)
    return tmp

def fetch_bi_grams(var):
    import numpy as np
    from gensim.models import Phrases
    from gensim.models.phrases import Phraser
    sentence_stream = np.array(var.str.split())
    bigram = Phrases(sentence_stream, min_count=5, threshold=10, delimiter=",")
    trigram = Phrases(bigram[sentence_stream], min_count=5, threshold=10)
    bigram_phraser = Phraser(bigram)
    trigram_phraser = Phraser(trigram)
    bi_grams = list()
    tri_grams = list()
    for sent in sentence_stream:
        bi_grams.append(bigram_phraser[sent]) #???
        tri_grams.append(trigram_phraser[sent])
    return bi_grams, tri_grams
    
def corpus_dic_creator(txt):
    import gensim.corpora as corpora
    id2word = corpora.Dictionary(txt)
    corpus = [id2word.doc2bow(text) for text in txt]
    return id2word, corpus

def compute_coherence_values(dictionary, corpus, k, a, b,text):
    import gensim
    from gensim.models import CoherenceModel
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=42,
                                           iterations =50,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=text, dictionary=dictionary, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

def hypertuning_lda(corpus,kt,alpha,beta, dictionary,text):
    import pandas as pd
    import numpy as np
    import gensim
    from tqdm import tqdm
    topics=list(np.arange(kt[0],kt[1],kt[2]))
    a_list=list(np.arange(alpha[0],alpha[1],alpha[2]))
    a_list.append("symmetric")
    a_list.append("asymmetric")
    b_list=list(np.arange(beta[0],beta[1],beta[2]))
    b_list.append("symmetric")
    grid = {}
    grid['Validation_Set'] = {}
    num_of_docs = len(corpus)
  #  corpus_sets = [gensim.utils.ClippedCorpus(corpus, num_of_docs*c_value), corpus]
   # corpus_title = ['{}Corpus'.format(c_value), 'all Corpus']
    model_results = {
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
    timelen=len(topics)*len(a_list)*len(b_list)
    if __name__ == '__main__':  
        with tqdm(total=timelen) as pbar:
            for k in topics:
                for a in a_list:
                    for b in b_list:
                        cv=compute_coherence_values(corpus=corpus,dictionary=dictionary,k=k,a=a,b=b,text=text)
                        model_results['Topics'].append(k)
                        model_results['Alpha'].append(a)
                        model_results['Beta'].append(b)
                        model_results['Coherence'].append(cv)
                        pbar.update(1)
    return pd.DataFrame(model_results)

def lda_fun(the_data,n, dictionary, corpus):
    import gensim
    from gensim.corpora.dictionary import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    import gensim.corpora as corpora
    import matplotlib.pyplot as plt
    from kneed import KneeLocator
    
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus=corpus, num_topics=n, id2word=dictionary, iterations=50, passes=15,
        random_state=123)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)
        
    #compute Coherence Score using c_v
    coherence_model_lda = CoherenceModel(
        model=ldamodel, texts=the_data,
        dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    
    c_scores = list()
    for word in range(1, n):
        print (word)
        ldamodel = gensim.models.ldamodel.LdaModel(
            corpus, num_topics=word, id2word=dictionary, iterations=10, passes=5,
            random_state=123)
        coherence_model_lda = CoherenceModel(model=ldamodel, texts=the_data,
                                              dictionary=dictionary,
                                              coherence='c_v')
        c_scores.append(coherence_model_lda.get_coherence())

    
    x = range(1, n)
    #https://pypi.org/project/kneed/
    kn = KneeLocator(x, c_scores,
                      curve='convex', direction='increasing')
    opt_topics = kn.knee
    print ("Optimal topics is", opt_topics)
    plt.plot(x, c_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    return 0

import pandas as pd

# RUN the chunk above first

df=pd.read_csv(r"D:\CU\spring 2022\nlp\Roku\amazon_OS10.csv") # read in the dataframe here

# text cleaning
df["bd_cleaned"]=clean_txt(df.body,5)

df["sw_removed"]=df["bd_cleaned"].apply(lambda x: my_stop_words(x,new=["roku","tv","amazon","ultra","use","would","device","one"]))

df["stemmed"]=df["sw_removed"].apply(my_stem)


# below here is what you can change as parameters
new_sw=["roku","tv","amazon","ultra","use","would","device","one"]
df["stemmed2"]=df["stemmed"].apply(lambda x: my_stop_words(x,new=new_sw))

# building bigrams and trigrams
bigram_text=fetch_bi_grams(df["stemmed2"])[0]  #don't change
trigram_text=fetch_bi_grams(df["stemmed2"])[1] #don't change
monogram_text=[i.split() for i in df["stemmed2"]] #don't change

# here you can change the text_to_use to the "bigram_text" to "trigram_text" or "monogram_text" in the bracket
# if you change the text_to_use, run all lines below
text_to_use=trigram_text
corpus=corpus_dic_creator(text_to_use)[1] # run as it is, creating essential object for further LDA model
dic=corpus_dic_creator(text_to_use)[0] # run as it is, creating essential object for further LDA model

# change n= as topic number
n=3
lda_fun(the_data=text_to_use,n=n, dictionary=dic, corpus=corpus)
# if you find a lot of repeating and meaningless words, put it into the new_sw list above and rerun the codes below that line
    