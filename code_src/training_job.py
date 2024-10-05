from typing import Dict, List, Tuple
import pandas as pd
from transformers import TrainingArguments
import os
from datasets import Dataset
from code_src.transfomers_src.SuperTrainer import SuperTrainer, EvaluateOnTrainSetCallback, complete_metrics
from code_src.transfomers_src import TransformersClasses as TC
from code_src.base.BaseClasses import TotalExperimentConfiguration, DataConfig, TrainerConfig, TrainingArgumentsConfig
from code_src.transfomers_src.utils import dump_configuration_to_json, add_predictions_n_probabilities_df
from code_src.dataset_handler_src.data_handler import DataHandler, DataSplitter
from tqdm import tqdm

############################################################################################################

import logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


META_MODEL_CONFIG = {'roberta': TC.RoBERTaBertTransformer,
                     'deberta': TC.DeBERTaBertTransformer,
                     'bert': TC.BertTransformer,
                     'distilbert': TC.DistilBertTransformer}

def dump_best_model(trainer,output_path, testset_path, test_predictions):
    save_predictions_df(testset_path,test_predictions, output_path )
    trainer.save_model(os.path.join(output_path, 'best_model')) 
    trainer.state.save_to_json( os.path.join(output_path, 'best_model','trainer_state.json'))

def save_predictions_df(dataset_path, predictions, 
                        experiment_output_path, file_name = 'test_set_predictions.xlsx'):
    df = pd.read_excel(dataset_path)
    df = add_predictions_n_probabilities_df(predictions, df)
    df.to_excel(os.path.join(experiment_output_path, file_name))


def train_model(model, trainer_config:TrainerConfig, training_arguments:TrainingArgumentsConfig, 
                tokenized_sets:Dict[Dataset], metrics) -> Tuple[SuperTrainer, Dict[str, float], Dict[str, float]]:
    
    LOGGER.info(f'Starting training process')
    trainer = SuperTrainer(model=model,
                           args=training_arguments,
                           train_dataset=tokenized_sets['train'],
                           eval_dataset=tokenized_sets['validation'],
                           compute_metrics=metrics,
                           loss_type=trainer_config.loss_type,
                           class_weights=trainer_config.use_class_weights,
                           focal_alpha = trainer_config.focal_alpha,
                           focal_gamma = trainer_config.focal_gamma)
    
    LOGGER.info('----using callback---')
    trainer.add_callback(EvaluateOnTrainSetCallback(trainer)) 

    trainer.train(resume_from_checkpoint=trainer_config.resume_from_checkpoint )
    LOGGER.info('Done fine-tuning process')
    
    pred = trainer.predict(tokenized_sets['test'])
    test_scores = metrics(pred)
    LOGGER.info(f'Test scores: {test_scores}')
    return trainer, pred, test_scores


def process_experiment_data(data_config:DataConfig, tokenizer:TC.SuperTokenizer,
                            train_path:str, test_path:str, validation_path:str=None) -> Tuple[Dict[str, Dataset], List[float]]:
    
    label_column = train_dataHandler.label_column
    train_dataHandler = DataHandler(train_path, data_config)
    test_dataHandler = DataHandler(test_path, data_config)
    
    if validation_df is None:
        train_df, validation_df = DataSplitter(data_config).split_data(train_dataHandler.dataset)
    else:
        validation_df = DataHandler(validation_path, data_config).dataset
        train_df = train_dataHandler.dataset
        
    LOGGER.info('train size is: ' + str(train_df[label_column].value_counts()))
    LOGGER.info('validation size is: ' + str(validation_df[label_column].value_counts()))
    LOGGER.info('test size is: ' + str(test_dataHandler.dataset[label_column].value_counts()))

    tokenized_train = tokenizer.process_df_to_tokenized_dataset(train_df)
    tokenized_validation =  tokenizer.process_df_to_tokenized_dataset(validation_df)
    tokenized_test =  tokenizer.process_df_to_tokenized_dataset(test_dataHandler.dataset)
    
    tokenized_sets = {'train': tokenized_train, 'validation': tokenized_validation, 'test': tokenized_test}

    return tokenized_sets, train_dataHandler.class_weights

def create_training_argument_object(config : TrainingArgumentsConfig) -> TrainingArguments:
    training_args = TrainingArguments(output_dir=config.training_output_path,
                                      evaluation_strategy="epoch",  # Train
                                      logging_strategy='epoch',  # Validation
                                      save_strategy="epoch",
                                      save_total_limit=config.save_total_limit,
                                      log_level='debug',  # 'passive'
                                      num_train_epochs=config.num_epochs,
                                      per_device_train_batch_size=config.batch_size,
                                      per_device_eval_batch_size=config.batch_size,
                                      gradient_accumulation_steps=config.gradient_accumulation_steps,
                                      learning_rate=config.learning_rate,
                                      lr_scheduler_type=config.lr_scheduler_type,
                                      load_best_model_at_end=True,
                                      metric_for_best_model=config.metric_for_best_model,
                                      greater_is_better=config.greater_is_better,
                                      torch_compile=True,
                                      warmup_steps = config.warmup_steps,
                                      fp16 = config.fp_16,
                                      gradient_checkpointing=config.gradient_checkpointing)
    return training_args

def train_and_predict(experiment_config: TotalExperimentConfiguration, 
                      trainset_path: str, testset_path: str, validation_path: str = None) -> SuperTrainer:
    
    LOGGER.info('Loading transformer model')
    transformer_config = experiment_config.transformer_config
    model = META_MODEL_CONFIG[transformer_config.bert_type](**vars(transformer_config)) #TODO
    
    LOGGER.info('Loading transformer model')
    tokenizer =  TC.SuperTokenizer(experiment_config.tokenizer_config)
    
    LOGGER.info('Processing datasets')
    tokenized_sets, train_classweights = process_experiment_data(experiment_config.data_config,
                                                                 tokenizer,
                                                                 trainset_path, testset_path, validation_path)
    
    class_weights_variable = experiment_config.trainer_config.use_class_weights
    if isinstance(class_weights_variable, bool) and class_weights_variable is True:
        experiment_config.trainer_config.use_class_weights = train_classweights

    training_args = create_training_argument_object(experiment_config.training_argumnets)
    trainer, test_predictions, test_scores = train_model(model.model,
                                                         trainer_config = experiment_config.trainer_config,
                                                         training_arguments=training_args,
                                                         metrics=complete_metrics,
                                                         tokenized_sets=tokenized_sets)
    LOGGER.info('Backing up results and best model')
    experiment_config.best_checkpoint = trainer.state.best_model_checkpoint
    experiment_config.test_scores = test_scores
    experiment_output_path = experiment_config.training_argumnets.training_output_path
    dump_best_model(trainer,experiment_output_path, testset_path,test_predictions)
    dump_configuration_to_json(experiment_config)

    LOGGER.info('Done Experiment')
    return trainer


if __name__ == "__main__":
    pass 