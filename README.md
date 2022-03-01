# MTL-ABSum

**This code is for the paper: *A Multi-Task Learning based Method for Text Summarization with Cross-Task Attention***

**Python Version**: This code is in Python3.6

**Package Requirements**:
```
spacy==2.2.4  
pyconll==2.3.1  
torch==1.5.0  
numpy==1.18.4  
rouge_metric==1.0.1  
gensim==3.8.3  
pyrouge==0.1.3  
scikit_learn==0.24.1  
transformers==4.5.0
```

# Data Preparation For MTL-ABSum

We use the CNN/DailyMail dataset for the summarization task, which consists of news on different topics. We use the Wall Street Journal (WSJ), Penn Treebank (PTB), and the AG News dataset to train the POS tagging, the dependency parsing, and the text categorization task, respectively.

## Prepare Datasets and Process Data
Download above datasets from public official web and put it to *dataset* folder, run:
```
cd datasets
python -u dealWSJ.py
python -u dealconll.py
python -u dealclass.py
python -u dealcnndm.py
```
Then, we will to convert word to numbers. In this files, you will be change some paramerters in line 209: 
```
dataset= xxx
```
run:
```
python -u convertData2nums.py
```
In the future, we will upload our processed data. 

# Model Training 

## step 1:  
To train the POS task model, run:
```
python -u posTask.py
```

## step 2:  
To train the Dependency Parsing task model, run:
```
python -u dpTask.py
```
After the dp task training finished, we will infer the dependency arc in each sentence of AG and CNN/DailyMail datasets, run:
```
python -u  runinfer.py 
```

## step 3:  
To train the Text categorization task model, run:
```
python -u classTask.py
```

## step 4:  
To train the Text Summarization model,you must first change the parameter in *[config.py](config.py)*: **is_coverage = False** , run:
```
python -u train.py 
```
When the training finished, you can select one low loss model and put in *model/cnndm-coverage* folder, and change the parameter in *[config.py](config.py)*: **is_coverage = True** to continue training.  

# Model Evaluation 

After the training finished, run
```
python -u run.py
```
## Notes:
 1. All model will be saved in the *model* folder
 2. All results will be save in the *results* folder
 3. The ROUGE results will be save in *rouge_results* folder