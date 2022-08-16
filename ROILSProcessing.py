"""
Text mining processing and analysis tool specifically designed for exported ROILS reports in xlsx format

Created on 08/16/2022
@author: Ryan Price
"""

import pandas as pd
import glob
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


####read in all data
event_data = pd.DataFrame()
processed_data = pd.DataFrame()
for f in glob.glob("O:/Research/Data mining ROILS/LO event detail list_All_08152022_RP.xlsx"):
    df = pd.read_excel(f)
    event_data = event_data.append(df,ignore_index=True,sort=False)

##PreProcess Narrative Summary and create new column with only narrative text
event_data['Narrative_txt'] = event_data.Narrative.str.partition("WHAT HAPPENED:")[2]
event_data['Narrative_txt'] = event_data.Narrative_txt.str.replace("(DOSE DISCREPANCY:).*","")


def preprocess(Narrative):
    custom_stopwords = ["rtts","happened","realized","initial","jk","jh","u","hz","mk","mp","gb","dkg","gw","kk","fs","rk","cd","rk","lmb","dg","ng","md","called"]
    stpwrd = stopwords.words('english')
    stpwrd.extend(custom_stopwords)
    Narrative=str(Narrative)
    Narrative = Narrative.lower() 
    cleanr = re.compile('<.*?>')
    Narrative = re.sub(cleanr, '', Narrative)
    #Narrative=re.sub(r'http\S+', '',Narrative)
    #Narrative = re.sub('[0-9]+', '', Narrative)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(Narrative)
 #   cv = CountVectorizer(lowercase=True,stpwrd,ngram_range = (1,2),tokens)
    filtered_words = [w for w in tokens if not w in stpwrd]
 #   stem_words=[stemmer.stem(w) for w in filtered_words]
  #  lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)


processed_data['processed_narrative']=event_data['Narrative_txt'].map(lambda s:preprocess(s))
