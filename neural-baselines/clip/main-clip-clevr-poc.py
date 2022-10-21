import torch
import pandas
import json
import argparse
import logging
import os
import cv2
import pickle

import numpy as np
import torch.nn as nn

from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoConfig
from transformers import Trainer, TrainingArguments


import albumentations as A

from transformers.modeling_outputs import SequenceClassifierOutput

from datasets import load_dataset, load_metric, DownloadConfig, load_from_disk, DatasetDict

from sklearn import metrics



class ClipClassification(nn.Module):
    def __init__(self, device, checkpoint, num_labels, outdim):
        super(ClipClassification,self).__init__()
        
        self.device = device
        self.num_labels = num_labels
        self.outdim = outdim

        self.model = CLIPModel.from_pretrained(checkpoint)
        self.model.to(self.device)
        self.dropout = nn.Dropout(0.1)
        #self.classifier = nn.Linear(outdim*2, num_labels) # load and initialize weights
        self.classifier = nn.Linear(outdim, num_labels) # load and initialize weights


    # define a function that returns the tensor of a specific constraint type
    def get_tensor(constraint_type):

        return torch.flatten(constraint_types_tensor[constraint_type])
   
        
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None, constraint_type=None):
        #Extract outputs from the body
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        constraint_type_list = constraint_type.tolist()
        constraint_type_embedding = list(map(ClipClassification.get_tensor, constraint_type_list))
        constraint_type_embedding = torch.stack([x for x in constraint_type_embedding], dim=0).to(self.device)
        #Add custom layers
        
        
        
        text_emb = outputs['text_embeds']    #8x512
        image_emb = outputs['image_embeds']  #8x512
        #emb = torch.cat([text_emb,image_emb], dim=1)   #8x1024
        emb = torch.cat([text_emb,image_emb,constraint_type_embedding], dim=1)  
        
        emb = self.dropout(emb)


        logits = self.classifier(emb) # calculate losses
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        hidden = outputs['text_model_output']['last_hidden_state']
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden,attentions=None)
        
        

## CONFIGURATION

data_folder = '/home/marjan/code/clevr-poc/data'
constraint_types_tensor_file_path = os.path.join(data_folder, 'constraint_types_tensor.pickle')

model_path = "openai/clip-vit-base-patch32"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
if torch.cuda.is_available():
    torch.cuda.empty_cache()

train_batch_size = 8
eval_batch_size = 8
num_workers = 8
pin_memory=8
gradient_accumulation=4
epochs = 20

max_length = 42

dropout = 0.1



## get constraints tensors

with open (constraint_types_tensor_file_path, 'rb') as f:
    constraint_types_tensor = pickle.load(f)
    
    
    
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler()])

logging.info("Loading dataset")

dl_config = DownloadConfig(resume_download=True, num_proc=4)
import ipdb; ipdb.set_trace()
logging.info('Loading training data')
dataset_train = load_dataset('clevr-poc-loader.py',
                       name='clevr-poc',
                       download_config=dl_config,
                       split='train[:]')
logging.info('Loading validation data')
dataset_val = load_dataset('clevr-poc-loader.py',
                       name='clevr-poc',
                       download_config=dl_config,
                       split='validation[:]')
logging.info('Loading test data')
dataset_test = load_dataset('clevr-poc-loader.py',
                       name='clevr-poc',
                       download_config=dl_config,
                       split='test[:]')

logging.info('Dataset loaded')

dataset = DatasetDict({
  'train':dataset_train,
  'validation':dataset_val,
  'test':dataset_test
})

logging.info('Loading CLIP')
model_path = "openai/clip-vit-base-patch32"

#TODO convert CLEVR images offline
extractor = CLIPProcessor.from_pretrained(model_path)

def transform_tokenize(e):
    e['image'] = [image.convert('RGB') for image in e['image']]
    return extractor(text=e['question'],
                               images=e['image'],
                               truncation=True, 
                               #padding=True)
                               padding="max_length", max_length=42)
    
    
    
logging.info('Transforming dataset')
dataset = dataset.map(transform_tokenize, batched=True, num_proc=1)






metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[:-1], axis=-1)[0]
    return metric.compute(predictions=predictions, references=labels)

dim_1 = 512
dim_2 = 512
dim_3 = constraint_types_tensor[0].shape[0]*constraint_types_tensor[0].shape[1]
outdim = dim_1 + dim_2 + dim_3
model = ClipClassification(device=device, checkpoint=model_path, num_labels=16, outdim=outdim)







logging.info("Creating trainer")
training_args = TrainingArguments("test_trainer",
                                    num_train_epochs=epochs,
                                    per_device_train_batch_size=train_batch_size,
                                    per_device_eval_batch_size=eval_batch_size,
                                    fp16=True if device == 'cuda' else False,
                                    dataloader_num_workers=num_workers ,
                                    dataloader_pin_memory=pin_memory,
                                    gradient_accumulation_steps=gradient_accumulation,
                                    save_strategy='no',
                                    evaluation_strategy='epoch',
                                    eval_steps=1)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)




logging.info("Training model")
training_metrics = trainer.train()
logging.info(training_metrics)





predictions, labels, test_metrics = trainer.predict(dataset['test'])
y_true = dataset['test']['label']                                                                                                                 
y_pred = np.argmax(predictions[:-1], axis=-1)[0]                                                                                                    
confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15])                                                                                                            
print(confusion_matrix)
logging.info(test_metrics	
    
