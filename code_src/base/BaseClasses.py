from abc import ABC, abstractmethod
from transformers import PreTrainedModel
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict


############################################################################################################

@dataclass
class TransformerConfig:
    bert_type: str
    model_name_or_path: str
    task: str = 'classification'
    num_labels: int = 2
    ignore_mismatch: bool = True
    dropout: float = 0.1
    attention_dropout: float = 0.1
    classification_dropout:float = 0.1
    freeze_embedding: bool = False
    freeze_layers: int = None

@dataclass
class TokenizerConfig:
    model_name_or_path:str
    max_length:int = 512 
    truncate:bool = True
    padding:str = 'max_length'

@dataclass
class TrainerConfig:
    class_weights: Optional[List[float]] = None
    loss_type: str = 'cross_entropy'
    lr_schedulert_type: str = 'linear'
    resume_from_checkpoint: bool = False
    use_class_weights: Union[bool, List[float]] = False
    compute_metrics: Optional[List[str]] = None
    focal_alpha: float = None
    focal_gamma: float = None

@dataclass
class DataConfig:
    text_column: str = 'text'
    label_column: str = 'label'
    process_text: bool = False
    cluster_text: bool = False
    cluster_config: Dict = field(default_factory=lambda: {'embed_type': 'tf_idf',
                                                          'embed_model': None,
                                                          'cluster_params': {'algorithm': 'evoc',
                                                                             'k_cluster': None,
                                                                             'random_state': 42},
                                                          'handle_outliers': False}) 
    split_ratio: float = 0.25
    startify_column: str = 'label'
    split_random_state:int = 42

@dataclass
class TrainingArgumentsConfig:
    training_output_path: str = 'trainer_tmp_folder'
    save_total_limit: Optional[int] = None
    num_epochs: int = 8
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    lr_scheduler_type: str = 'linear'
    metric_for_best_model: str = 'f1'
    greater_is_better: bool = True
    warmup_steps: int = 0
    fp_16:bool = False
    gradient_checkpointing: bool = False

@dataclass
class TotalExperimentConfiguration:
    data_config: DataConfig
    transformer_config: TransformerConfig
    tokenizer_config: TokenizerConfig
    trainer_config: TrainerConfig
    training_argumnets: TrainingArgumentsConfig
    best_checkpoint: str = None
    test_scores: Dict = field(default_factory=dict)


############################################################################################################

class TransformerBase(ABC):
    def __init__(self,
                 config: TransformerConfig):
        
        self.bert_type = config.bert_type
        self.model = self.load_model(config.model_name_or_path)
        self.tokenizer = None
        self.task = config.task
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    @abstractmethod
    def load_model(self, model_name_or_path: str) -> PreTrainedModel:
        pass

    @abstractmethod
    def freeze_layers(self):
        pass

    @abstractmethod
    def freeze_embeddings(self):
        pass
    
    @abstractmethod
    def assert_tokenizer_and_model_size(self):
        pass
    
    @abstractmethod
    def add_token_list_to_model(self, tokenizer, token_list:str):
        self.tokenizer = tokenizer
        self.tokenizer.add_tokens(token_list)
        self.model.resize_token_embeddings(len(self.tokenizer))
        print('Resized Tokenizer: ' + str(len(self.tokenizer)))

