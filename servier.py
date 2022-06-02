from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
from pandarallel import pandarallel
pandarallel.initialize()
import dateutil.parser as dparser
import networkx as nx
import warnings
warnings.filterwarnings('ignore') # supress warnings due to some future deprications
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from networkx.readwrite import json_graph
import json


def get_df(file):
    return pd.read_csv(file)

def title_preprocessing(text):
    text = str(text)
    # remove punctuation
    remove_terms = punctuation + '0123456789'
    # tokenize and lower case text
    words = word_tokenize(text)
    tokens = [w.lower() for w in words if w.lower() not in remove_terms]
    # remove english stopwords
    stopw = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopw]
    # remove words less than one letters
    tokens = [word for word in tokens if len(word)>1]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    return set(tokens)

def journal_preprocessing(text):
    text = str(text)
    # remove characters that are not in [^A-Za-záàâäãåçéèêëôï] from the ends of words
    text = re.split(r"[^A-Za-záàâäãåçéèêëôï]", text.strip())
    text = ' '.join(text)
    # remove punctuation
    remove_terms = punctuation + '0123456789'
    # tokenize and lower case text
    words_tokenized = word_tokenize(text)
    tokens = [w for w in words_tokenized if w.lower() not in remove_terms]
    # remove english stopwords
    stopw = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopw]
    # remove words less than one letters
    tokens = [word for word in tokens if len(word)>2]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # create a string from tokens
    tokens = ' '.join(tokens)
    return tokens

def date_format(date):
    return dparser.parse(date,fuzzy=True).strftime("%d/%m/%y")

def df_preprocessing(df):
    for t in df.columns.values.tolist():
        if 'title' in t:
            df['title_set'] = df[t].parallel_apply(title_preprocessing)
            break
    df['journal'] = df['journal'].parallel_apply(journal_preprocessing)
    df['date'] = df['date'].parallel_apply(date_format)
    return df

def all_drugs(df_drugs):
    drugs = set()
    for i in df_drugs.index:
        drugs.add(df_drugs.iloc[i]["drug"].lower())
    return drugs 

def add_drugs(df, df_drugs):
    drugs = all_drugs(df_drugs)
    for i in df.index:
        inter = df.iloc[i]['title_set'].intersection(drugs)
        if len(inter) == 0:
            df.at[i, 'drug'] = ''
        else:
            df.at[i, 'drug'] = list(inter)[0]
    df.dropna(inplace=True)
    df.drop(df.index[df['drug'] == ''], inplace=True)
    df.reset_index(drop=True, inplace = True)
    return df

def build_digraph(df1, df2):
    G = nx.DiGraph()
    df_list = [df1, df2]
    for df in df_list:
        for i in df1.index:
            G.add_edge(df1.iloc[i]['drug'], df1.iloc[i]['journal'])
            nx.set_edge_attributes(G, {(df1.iloc[i]['drug'], df1.iloc[i]['journal']): {"date": df1.iloc[i]['date']}})
            nx.set_node_attributes(G, {df1.iloc[i]['drug']: "drug", df1.iloc[i]['journal']: "journal"}, name="name")
            try:
                G.add_edge(df1.iloc[i]['drug'], df1.iloc[i]['scientific_title'])
                nx.set_node_attributes(G, {df1.iloc[i]['scientific_title']: "clinical trial"}, name="name")
            except:
                G.add_edge(df1.iloc[i]['drug'], df1.iloc[i]['title'])
                nx.set_node_attributes(G, {df1.iloc[i]['title']: "pubmed"}, name="name")
    return G

def info_graph(G):
    print(nx.info(G))

def draw_graph(G):
    options = {
    'node_color': 'blue',
    'node_size': 300,
    'width': 2,
    }
    plt.figure(figsize = (8,5))
    nx.draw(G, with_labels = True, **options)
    plt.show()    

def output(G):
    return json_graph.node_link_data(G)

def save_json(G):
    json_data = output(G)
    with open('output.json', 'w') as json_file:
        json.dump(json_data, json_file)

def max_degree_journal(G):
    journal_degrees = {}
    for k, v in nx.get_node_attributes(G, "name").items():
        if v == 'journal':
            journal_degrees[k] = G.degree[k]
    return max(journal_degrees, key=journal_degrees.get)

def max_degree_drug(G):
    drug_degrees = {}
    for k, v in nx.get_node_attributes(G, "name").items():
        if v == 'drug':
            drug_degrees[k] = G.degree[k]
    return max(drug_degrees, key=drug_degrees.get)

def max_degree_pubmed(G):
    pubmed_degrees = {}
    for k, v in nx.get_node_attributes(G, "name").items():
        if v == 'pubmed':
            pubmed_degrees[k] = G.degree[k]
    return max(pubmed_degrees, key=pubmed_degrees.get)

def max_degree_graph(G):
    return max(max_degree_drug(G), max_degree_journal(G), max_degree_pubmed(G))