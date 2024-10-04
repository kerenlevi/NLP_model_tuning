import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from playground.code_src.transfomers_src.SuperTrainer import complete_metrics
import torch
import os 


def predict_and_extract_predictions(pipeline: TextClassificationPipeline, text_list: list):
    predictions = pipeline(text_list, padding=False, truncation=True, max_length=512, batch_size=1)
    class_result = []
    probabilities = []
    for pred in predictions:
        class_result.append(int(pred['label'].split('_')[-1]))
        probabilities.append(pred['score'])
    return class_result, probabilities


def get_text_and_label_lists(df_path, text_column:str = 'text', label_column:str='label'):
    df = pd.read_excel(df_path)
    text_list = list(df[text_column].astype(str))
    label_list = list(df[label_column].astype(int))
    return text_list, label_list


def predict_in_inference(model_path, tokenizer_path, 
                         dataset_path = None,  data_set = None, 
                         text_column = 'text', label_column = 'label',
                         output_path = None,):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model.resize_token_embeddings(len(tokenizer))
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

    if dataset_path:
        data_set = pd.read_excel(dataset_path)

    text_list = list(data_set[text_column].astype(str))
    class_result, probabilities = predict_and_extract_predictions(pipeline, text_list)
    if label_column in data_set.columns:
        metrics = complete_metrics(data_set[label_column], class_result)
        print(metrics)
        
    data_set['probability'] = probabilities
    data_set['prediction'] = class_result
    
    if output_path:
        data_set.to_excel(output_path)
    return data_set


def calculate_for_a_set(pipeline, text_list, label_list, device):
    class_result, _ = predict_and_extract_predictions(pipeline, text_list)
    metrics = complete_metrics(label_list, class_result)
    return metrics


def evaluate_all_checkpoints(tokenizer_path:str, checkpoints_path:str, 
                             train_path:str, validation_path:str, test_path:str,
                             text_column:str = 'text', label_column:str='label'):
    
    metrics_train, metrics_val, metrics_test = {}, {}, {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_text, train_label =  get_text_and_label_lists(train_path, text_column, label_column)
    validation_text, validation_label = get_text_and_label_lists(validation_path, text_column, label_column)
    test_text, test_label = get_text_and_label_lists(test_path, text_column, label_column)

    for checkpoint_path in checkpoints_path:
        _, fold_location = os.path.split(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
        
        metrics_train[fold_location] = calculate_for_a_set(pipeline, train_text, train_label,device)
        metrics_val[fold_location] = calculate_for_a_set(pipeline, validation_text, validation_label,device)
        metrics_test[fold_location] = calculate_for_a_set(pipeline, test_text, test_label,device)
        
    return metrics_train, metrics_val, metrics_test

