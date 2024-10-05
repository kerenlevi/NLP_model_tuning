
from sklearn.model_selection import train_test_split 
from sklearn.utils import class_weight
import pandas as pd
from code_src.base.BaseClasses import  DataConfig
from typing import List
from code_src.dataset_handler_src.nlp_clustering_pipeline import EmbeddText, ClusteringText

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
"""

class DataSplitter():
    def __init__(self,
                 config: DataConfig = DataConfig):
        self.startify_column = config.startify_column
        self.split_ratio =config.split_ratio
        self.random_state = config.split_random_state
    
    def split_data(self, data):
        splitted_first, splitted_second = train_test_split(data, test_size=self.split_ratio, 
                                                           stratify=data[self.columns_for_split],
                                                           random_state=self.random_state)
    
        return splitted_first, splitted_second


class DataHandler():
    def __init__(self,
                 dataset_path:str, 
                 config: DataConfig = DataConfig):
        
        self.paths = dataset_path
        self.dataset = pd.read_excel(dataset_path) if dataset_path.endswith('.xlsx') else pd.read_csv(dataset_path)
        self.text_column = config.text_column
        self.label_column = config.label_column
        self.classes = self.train_set[self.label_column].unique() if self.label_column is not None else None
        self.process_text = config.process_text
        self.cluster_text = config.cluster_text
        self.cluster_config = config.cluster_config
        self._post_init()
        
    def _post_init(self):
        self.class_weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                               classes = self.classes, 
                                                               y=self.dataset[self.label_column]) if self.classes is not None else None
        
        
        self.processed_data = self.process_data(self.dataset[self.text_column]) if self.process_text is True else None
        self.embedd_and_cluster(self.cluster_config['embed_type'], 
                                self.cluster_config['cluster_params'], self.cluster_config['handle_outliers']) if self.cluster_text is True else None
    
    def process_data(self):
        pass

    def embedd_and_cluster(self, embed_type:str, cluster_params:dict, embed_model:str = None, handle_outliers:bool = False) ->List:
        embedd_text = EmbeddText(self.data, self.text_column, embed_type, embed_model)
        clusters = ClusteringText(embedd_text, cluster_params)
        self.clusters = clusters.cluster_labels[cluster_params['algorithm']]
        self.cluster_column_name = f'{embed_type}_{cluster_params["algorithm"]}_cluster'
        print(f'Data embedded and clustered successfully using: {self.cluster_column_name}')
        if handle_outliers is True:
            updated_clusters =  self.remove_outlier_clusters(self.dataset, self.cluster_column_name)
        return clusters , updated_clusters

    def update_startify_column_with_label_column(self, startify_column:str = 'stratify_col'):
        self.startify_column =startify_column
        self.dataset[self.startify_column] = self.dataset[self.label_column].astype(str) + "_" + self.dataset[self.cluster_column_name].astype(str)
    
    def remove_outlier_clusters(self, cluster_column_name:str, threshold:int = 5, outlier_string:str='other') -> List:
        """This function removes clusters with less than threshold samples and assigns them to a new cluster called 'other'"""
        value_counts = self.dataset[cluster_column_name].value_counts() 
        single_occurrences = value_counts[value_counts <=threshold].index.tolist()
        updated_clusters = self.dataset[cluster_column_name].apply(lambda x: outlier_string if x in single_occurrences else x).to_list()
        return updated_clusters

