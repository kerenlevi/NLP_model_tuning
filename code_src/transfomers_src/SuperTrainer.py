from transformers import TrainingArguments, Trainer, TrainerCallback, get_linear_schedule_with_warmup
import torch
from torch import nn
import torch.nn.functional as F
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


def complete_metrics(model_pred):
    """ Trainer Prediction output is an PredictionOutput array with 2 attributes : (1) predictions  (2) label_ids """
    logits = model_pred.predictions.argmax(-1)
    labels = model_pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, logits, average='binary')
    acc = accuracy_score(y_pred=logits, y_true=labels)
    tn, fp, fn, tp = confusion_matrix(labels, logits).ravel()
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Gamma: Higher values of gamma put more focus on hard-to-classify examples.
        If recall is high but precision is low, it might be that the model is focusing too much on hard examples and making more false positives.
        Try reducing gamma.
    Alpha: This is used to balance the importance of positive/negative classes. 
        Adjusting alpha can help balance precision and recall if your dataset is imbalanced.
        
    try: (alpha=0.75, gamma=1.5)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = (alpha * (1-pt)**gamma * ce_loss).mean()
    return focal_loss


class SuperTrainer(Trainer):
    def __init__(self, *args, class_weights=None,loss_type=None,
                 lr_schedulert_type=None, 
                 focal_alpha=None, focal_gamma=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.loss_type = loss_type
        self.lr_schedulert_type =lr_schedulert_type
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                if self.loss_type == 'focal':
                    assert self.focal_alpha is not None and self.focal_gamma is not None, 'Please provide alpha and gamma values for focal loss'
                    loss = focal_loss(logits, labels)
                elif self.loss_type == 'class_weight':
                    loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weights)).to(device, dtype=torch.float32)
                    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                else:
                    loss = F.cross_entropy(logits, labels)
                return (loss, outputs) if return_outputs else loss
    
    # https://discuss.huggingface.co/t/how-do-use-lr-scheduler/4046/8
    # def create_optimizer_and_scheduler(self, num_training_steps):
    #     self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    #     if self.lr_schedulert_type=='polynomial':
    #         self.lr_scheduler = get_polynomial_decay_schedule_with_warmup(self.optimizer, 0, num_training_steps, power=2)
            

class EvaluateOnTrainSetCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    def on_evaluate(self, args, state, control, **kwargs):
        control_copy = deepcopy(control)                
        callback_backup = self._trainer.pop_callback(EvaluateOnTrainSetCallback)  # to prevent recursive call of evaluation
        self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
        self._trainer.add_callback(callback_backup)  # restore the callback
        
        return control_copy
    

class EpochEvaluateTrainSet(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            callback_backup = self._trainer.pop_callback(EpochEvaluateTrainSet)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            self._trainer.add_callback(callback_backup)
            return control_copy