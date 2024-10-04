from transformers import BertModel, AutoTokenizer, AutoModelForSequenceClassification
from code_src.base.BaseClasses import  TokenizerConfig, TransformerConfig, TransformerBase

from datasets import Dataset
import torch

############################################################################################################

    # dropout = model_configuration.get('dropout', 0.1)
    # attention_dropout = model_configuration.get('attention_dropout', 0.1)
    # classification_dropout = model_configuration.get('seq_classification_dropout', 0.2)


class DistilBertTransformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.num_labels = config.num_labels
        self.ignore_mismatch = config.ignore_mismatch
        self.model = self.load_model(config.model_name_or_path, config.num_labels, ignore_mismatch=config.ignore_mismatch)
        if config.freeze_embedding: self.freeze_embeddings()
        self.freeze_layers(config.freeze_layers) 
        
    def load_model(self, model_name_or_path: str, num_labels:int = None, ignore_mismatch:bool = True) -> BertModel:
        if self.task =="classification":
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                      num_labels=num_labels,
                                                                      ignore_mismatched_sizes=ignore_mismatch).to(self.device, dtype=torch.float32)
        else:
            return BertModel.from_pretrained(model_name_or_path).to(self.device, dtype=torch.float32)

    def freeze_layers(self, num_layers: int = 1):
        print('First %s  layers are frozen' % num_layers)
        for layer in self.model.distilbert.transformer.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze_embeddings(self):
        print('Embedding layer frozen')
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False

class BertTransformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.num_labels = config.num_labels
        self.ignore_mismatch = config.ignore_mismatch
        self.model = self.load_model(config.model_name_or_path, config.num_labels, ignore_mismatch=config.ignore_mismatch)
        if config.freeze_embedding: self.freeze_embeddings()
        self.freeze_layers(config.freeze_layers) 
        
    def load_model(self, model_name_or_path: str, num_labels:int = None, ignore_mismatch:bool = True) -> BertModel:
        if self.task =="classification":
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                      num_labels=num_labels,
                                                                      ignore_mismatched_sizes=ignore_mismatch).to(self.device, dtype=torch.float32)
        else:
            return BertModel.from_pretrained(model_name_or_path).to(self.device, dtype=torch.float32)

    def freeze_layers(self, num_layers: int = 1):
        print('First %s  layers are frozen' % num_layers)
        for layer in self.model.distilbert.transformer.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze_embeddings(self):
        print('Embedding layer frozen')
        for param in self.model.distilbert.embeddings.parameters():
            param.requires_grad = False

class RoBERTaBertTransformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.num_labels = config.num_labels
        self.ignore_mismatch = config.ignore_mismatch
        self.model = self.load_model(config.model_name_or_path, config.num_labels, ignore_mismatch=config.ignore_mismatch)
        if config.freeze_embedding: self.freeze_embeddings()
        self.freeze_layers(config.freeze_layers) 
        
    def load_model(self, model_name_or_path: str, num_labels:int = None, ignore_mismatch:bool = True) -> BertModel:
        if self.task =="classification":
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                      num_labels=num_labels,
                                                                      ignore_mismatched_sizes=ignore_mismatch).to(self.device,
                                                                                                                  dtype=torch.float32)
        else:
            return BertModel.from_pretrained(model_name_or_path).to(self.device, dtype=torch.float32)

    
    def freeze_layers(self, num_layers: int = 1):
        print('First %s  layers are frozen' % num_layers)
        for layer in self.model.roberta.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze_embeddings(self):
        print('Embedding layer frozen')
        for param in self.model.roberta.embeddings.parameters():
            param.requires_grad = False

class DeBERTaBertTransformer(TransformerBase):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.num_labels = config.num_labels
        self.ignore_mismatch = config.ignore_mismatch
        self.model = self.load_model(config.model_name_or_path, config.num_labels, ignore_mismatch=config.ignore_mismatch)
        if config.freeze_embedding: self.freeze_embeddings()
        self.freeze_layers(config.freeze_layers) 
        
    def load_model(self, model_name_or_path: str, num_labels:int = None, ignore_mismatch:bool = True) -> BertModel:
        if self.task =="classification":
            return AutoModelForSequenceClassification.from_pretrained(model_name_or_path,
                                                                      num_labels=num_labels,
                                                                      ignore_mismatched_sizes=ignore_mismatch).to(self.device, dtype=torch.float32)
        else:
            return BertModel.from_pretrained(model_name_or_path).to(self.device, dtype=torch.float32)

    def freeze_layers(self, num_layers: int = 1):
        print('First %s  layers are frozen' % num_layers)
        for layer in self.model.deberta.encoder.layer[:num_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

    def freeze_embeddings(self):
        print('Embedding layer frozen')
        for param in self.model.deberta.embeddings.parameters():
            param.requires_grad = False

############################################################################################################

class SuperTokenizer():
    def __init__(self, config: TokenizerConfig):
        self.tokenizer = self.load_model(config.model_name_or_path)
        self.max_length = config.max_length
        self.truncate = config.truncate
        self.padding = config.padding
        
    def load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(model_name_or_path)

    def tokenize(self, examples):
        return self.tokenizer(examples["text"], padding=self.padding, truncation=self.truncate, max_length=self.max_length)
    
    def process_df_to_tokenized_dataset(self, dataframe, text_column, label_column):
        dataframe[text_column] = dataframe[text_column].map(lambda x: str(x))
        dataframe[label_column] = dataframe[label_column].map(lambda x: int(x))
        
        set_for_tokenizer = Dataset.from_pandas(dataframe[[text_column, label_column]])
        tokenized_data = set_for_tokenizer.map(lambda batch: self.tokenize(batch), batched=True)
        
        return tokenized_data
