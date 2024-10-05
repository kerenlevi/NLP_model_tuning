from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight
import pandas as pd
from code_src.base.BaseClasses import  DataConfig
from typing import List
from sentence_transformers import SentenceTransformer
from transformers import BertModel, BertTokenizer
import numpy as np
import torch
from torchtext.vocab import GloVe
import evoc

"""
0_ Load data & Process
1_ embedd data 
    Init:
        Model to be used for embedding: Bert, Sentence-Transformer,TF-IDF etc 
        Input for Embedding: X = data_frame[text_column]
    Output: Y = embedd_data
        
2_ cluster data
3_ update label column
4_ split data

#TODO
# Random selection for variety of data [distribution of #sentences, #participants, #features, clusters, sender]
#Correltion between #participant & #sentences in IM for N&C 
#Sample only a section of suppressors (there are a lot )
# When clustering add another token of 1/0 for suppressors 
1) Token similarity (how to count number of similar tokens)
token distribution for similarity of texts?
Clustering over semantices 
Cluster over tokens 
sample from a cluster 
"""


class ClusteringText():    
    def __init__(self, vector_data:list, cluster_params:dict = {'algorithm':'evoc'}):
        self.vector_data = vector_data
        self.cluster_algorithm = cluster_params['algorithm']
        self.cluster_labels = {}
        
        if self.cluster_algorithm == 'k_means':
            k_clusters = cluster_params.get('k_clusters', 5)
            random_state = cluster_params.get('random_state', 42)
            self.k_means_cluster(k_clusters, random_state)
        elif self.cluster_algorithm == 'evoc':
            noise_level = cluster_params.get('noise_level', 0.5)
            base_min_cluster_size = cluster_params.get('base_min_cluster_size', 5)
            min_num_clusters = cluster_params.get('min_num_clusters', 4)
            approx_n_clusters = cluster_params.get('approx_n_clusters', None)
            return_duplicates = cluster_params.get('return_duplicates', False)
            self.evoc_cluster(noise_level, base_min_cluster_size, min_num_clusters, approx_n_clusters, return_duplicates)
    
    def k_means_cluster(self, k_clusters:int, random_state:int = 42) -> list:
        kmeans = KMeans(n_clusters=k_clusters, random_state=random_state, n_init='auto')
        kmeans.fit(self.vector_data)
        labels = kmeans.predict(self.vector_data)
        self.cluster_labels['k_means'] = labels
    
    def evoc_cluster(self, noise_level, base_min_cluster_size, min_num_clusters, approx_n_clusters, return_duplicates):
        clusterer = evoc.EVoC(noise_level=noise_level,
                              base_min_cluster_size=base_min_cluster_size,
                              min_num_clusters=min_num_clusters,
                              approx_n_clusters=approx_n_clusters,
                              return_duplicates=return_duplicates)
        
        cluster_labels = clusterer.fit_predict(self.vector_data)
        self.cluster_labels['evoc'] = cluster_labels
        # cluster_layers = clusterer.cluster_layers_ #  cluster granularity,
        # hierarchy = clusterer.cluster_tree_ # hierarchy of clusters across those layers
        # potential_duplicates = clusterer.duplicates_ # automatic duplicate (or very near duplicate) detection 
        
    def HDBSCAN(self):
        # Need to use Umap on embeddings 
        pass

class EmbeddText():
    def __init__(self, data_frame, text_column, embed_type:str = 'tf_idf', embed_model:str = None):
        self.data = data_frame
        self.text_column = text_column
        
        if embed_type == 'tf_idf':
            self.embedd_using_tf_idf()
        elif embed_type == 'sentence_transformer':
            self.embedd_using_sentence_transformer(embed_model)
        elif embed_type == 'bert':
            self.embedd_using_bert(embed_model)
        elif embed_type == 'glove':
            self.embedd_using_glove(embed_model)
        
    def embedd_using_tf_idf(self) -> List:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.data[self.text_column])
        self.tf_idf_embedding = X
    
    def embedd_using_sentence_transformer(self, model_name:str = 'sentence-transformers/all-MiniLM-L6-v2') -> List:
        model = SentenceTransformer(model_name)
        if 'e5-' in model_name:
            # X = self.data[self.text_column].apply(lambda text: model.encode(['query: ' + text], convert_to_numpy=True).flatten())
            X = model.encode(self.data[self.text_column].apply(lambda x: f'query: {x}').tolist(), convert_to_numpy=True) 
        else:
            X = model.encode(self.data[self.text_column].to_list(), convert_to_numpy=True) 
            # X = self.data[self.text_column].apply(lambda text: model.encode(text, convert_to_numpy=True).flatten())
        # X = np.vstack(X)
        self.sentence_transformer_embedding = X 

    def embedd_using_bert(self, model_name:str = 'bert-base-uncased') -> List:
        def get_cls_embedding(text, tokenizer, model):
            input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True, max_length=512)])
            with torch.no_grad():
                outputs = model(input_ids)
                cls_embedding = outputs[0][:, 0, :]
            return cls_embedding.flatten()
        
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        X = self.data[self.text_column].apply(lambda text: get_cls_embedding(text, tokenizer, model))
        X_bert = np.vstack(X)
        self.bert_embedding = np.vstack(X_bert)
        
    def embedd_using_glove(self, model_name:str ='6B', max_length = 100, embedding_dim = 100):
        def sentence_embedding(text):
            words = text.split()
            num_words = min(len(words), max_length)
            embedding_sentence = np.zeros((max_length, embedding_dim))
            for i in range(num_words):
                    word = words[i]
                    if word in embeddings.stoi:
                        embedding_sentence[i] = embeddings.vectors[embeddings.stoi[word]]
            return embedding_sentence.flatten()
        
        embeddings = GloVe(name=model_name, dim=embedding_dim)
        X = self.data[self.text_column].apply(lambda text: sentence_embedding(text))
        X_glove = np.vstack(X)
        self.glove_embedding = X_glove

class DataSplitter():
    def __init__(self,
                 config: DataConfig = DataConfig):
        self.startify_column = config.startify_column
        self.split_ratio =config.split_ratio
        self.random_state = config.random_state
    
    def split_data(self, data):
        splitted_first, splitted_second = train_test_split(data, test_size=self.split_ratio, 
                                                           stratify=data[self.columns_for_split],
                                                           random_state=self.random_state)
    
        return splitted_first, splitted_second