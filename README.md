# NLP_model_tuning

This library hold main steps in tuning  an NLP [in specific, transformer] model

### the following main steps are: 
1) loading and processing datasets (dataset_handler_src)
2) loading model, tokenizer and Trainer objects (modified to include various hyperparameter such as layer freezing)
3) Training a model and evaluation over a test-set 
4) inference code 


### Work in Progress 
Create a Grid Search library such that will
- Work in parallel way 
- will include convergance of loss across epochs 
