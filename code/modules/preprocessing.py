import spacy
from spacy.lang import en
from spacy import load
from spacy.lang.en import English

from nltk.tokenize import RegexpTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

def free_form_preprocessor(df, column_name, stops):
    """
    Free form text processor.
    Extracts words only, removes stop words, to lowercase, and strips white space.
    
    df : a pandas dataframe
    column_name : a name of a column in df
    stops : a list of stopwords; if "default" use union of nltk and sklearn stopwords
    
    returns processed text
    """
    
    if stops == "default":
        sw = set(stopwords.words("english"))
        cv = set(CountVectorizer(stop_words="english").get_stop_words())

        # union of stopwrods from nltk and CV extended with custom stopwords
        stops = list(set(cv | sw))
        
    tmp = df[column_name].map(RegexpTokenizer(r"\w+").tokenize)
    tmp = tmp.map(lambda x: " ".join(x))
    tmp = tmp.map(str.lower)
    tmp = tmp.map(lambda x: " ".join(w for w in x.split() if w not in stops))
    tmp = tmp.map(str.strip)
    
    return tmp

def free_form_counter(df, column_name):
    """
    Free form text word and character counter.
    Takes a df and column. Returns a word count and character count(including white space) of
    column.
    
    df : a pandas dataframe
    column_name : a name of column in df
    
    returns character count, word count
    """
    tmp_char_cnt = df[column_name].str.len()
    tmp_word_cnt = df[column_name].str.split().str.len()
    
    return tmp_char_cnt, tmp_word_cnt

def free_form_sent(df, column_name):
    """
    Freem form text sentiment analyzer.
    Analyzes sentiment of text returns a dataframe with scores as columns
    
    df : a pandas dataframe
    column_name : a name of a column in df
    
    returns four column dataframe of sentiments
    """
    sia = SentimentIntensityAnalyzer()

    sent_dict_df = df[column_name].map(lambda x: sia.polarity_scores(x))
    sent_df   = sent_dict_df.map(lambda x: [v for k, v in x.items()]).apply(pd.Series)
    
    return sent_df.set_axis(["NEGATIVE" + column_name, "NEUTRAL" + column_name, "POSITIVE" + column_name,         "COMPOUND" + column_name], axis = 1, inplace=True)

def lemmings(df, text_columns, pos = []):
    """
    
    """
    space_case = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
    
    df["text"] = ""
    
    # agglomerate/aggregate text features
    for col in text_columns:
        df["text"] += df[col]
        
    grammar = [space_case(x) for x in df["text"]]
    
    if pos != []:
        lemmus = [" ".join(token.lemma_ for token in doc if token.pos_ in pos) for doc in grammar]
    else:
        lemmus = [" ".join(token.lemma_ for token in doc) for doc in grammar]
    
    return lemmus