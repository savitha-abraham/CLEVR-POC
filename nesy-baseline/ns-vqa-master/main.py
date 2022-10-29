#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from IPython import get_ipython 
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from pathlib import Path
import os, json
ns_vqa_root = os.path.abspath('') # /home/savitha/Documents/clevr-poc/CLEVR-POC/nesy-baseline/ns-vqa-master
main_root = Path(ns_vqa_root).parents[1] #/home/savitha/Documents/clevr-poc/CLEVR-POC
DATA_FOLDER_NAME = "output-2000"


# In[ ]:


#PREPROCESS TRAIN

import os
shell = TerminalInteractiveShell.instance()
#%run tools/preprocess_questions.py \
#--input_questions_json=../data/raw/CLEVR_v1.0/questions/CLEVR_train_questions.json \
#--output_h5_file=../data/reason/clevr_h5_3000/clevr_train_questions_pre.h5 \
#--input_vocab_json=../data/reason/clevr_h5_3000/clevr_vocab_pre.json
run = f"python ./reason/tools/preprocess_questions.py         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}         --data_folder_name={DATA_FOLDER_NAME}         --vocab_flag=0" 
get_ipython().system('{run} ')


# In[ ]:


#PREPROCESS VAL
run = f"python ./reason/tools/preprocess_questions.py         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}         --data_folder_name={DATA_FOLDER_NAME}         --vocab_flag=1" 
get_ipython().system('{run} ')



# In[ ]:


#SAMPLING

run = f"python ./reason/tools/sample_questions.py         --n_questions_per_family 3         --input_question_h5 {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-train_questions.h5         --output_dir {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}" 
get_ipython().system('{run} ')




# In[ ]:


run = f"python ./reason/tools/run_train.py         --clevr_train_scene_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/scenes/training         --clevr_val_scene_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/scenes/validation         --checkpoint_every 100         --num_iters 400         --max_val_samples 50         --run_dir {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/model_pretrain         --clevr_train_question_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr_train_3questions_per_family.h5         --clevr_val_question_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-val_questions.h5         --clevr_constraint_scene_path {main_root}/clevr-poc-dataset-gen/environment_constraints         --clevr_vocab_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-vocab.json         --train_image_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/images/training         --val_image_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/images/validation         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}        --data_folder_name={DATA_FOLDER_NAME}" 
    
    
                                         
get_ipython().system('{run} ')


        


# In[ ]:


#Plot train, val loss - pretraining
import pickle
import matplotlib.pyplot as plt
import matplotlib

path_trainLoss = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME, 'train_loss_P.pickle')
path_valLoss = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME, 'val_loss.pickle')


with open(path_trainLoss, 'rb') as fp:
    train_loss_P = pickle.load(fp)
with open(path_valLoss, 'rb') as fp:
    val_loss = pickle.load(fp)

epoch_count = range(1, len(val_loss) + 1)
plt.figure(figsize=(10, 7))
plt.plot(epoch_count, train_loss_P, 'b-', color='orange')
plt.plot(epoch_count, val_loss, 'b-', color='green')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
path_plot = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME,'pretrain-plot.png')
plt.savefig(path_plot)
plt.show()


# In[ ]:


run = f"python ./reason/tools/run_train.py         --clevr_train_scene_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/scenes/training         --clevr_val_scene_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/scenes/validation         --checkpoint_every 100         --num_iters 310000         --run_dir {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/model_reinforce         --clevr_train_question_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-train_questions.h5         --clevr_val_question_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-val_questions.h5         --clevr_constraint_scene_path {main_root}/clevr-poc-dataset-gen/environment_constraints         --clevr_vocab_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-vocab.json         --train_image_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/images/training         --val_image_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/images/validation         --load_checkpoint_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/model_pretrain/checkpoint_best.pt         --reinforce 1         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}         --data_folder_name={DATA_FOLDER_NAME}" 
    
    
                                         
get_ipython().system('{run} ')



# In[ ]:


#Plot train, val loss/reward - epoch
# plot and save the train and validation line graphs
import pickle
import matplotlib.pyplot as plt
import matplotlib


path_trainLoss_R = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME, 'train_loss_R.pickle')
path_trainReward = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME, 'train_reward.pickle')
path_valAcc = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME, 'val_acc_R.pickle')

with open(path_trainReward, 'rb') as fp:
    train_reward = pickle.load(fp)
with open(path_valAcc, 'rb') as fp:
    val_Acc = pickle.load(fp)
with open(path_trainLoss_R, 'rb') as fp:
    train_loss_R = pickle.load(fp)

epoch_count = range(1, len(train_reward) + 1)
plt.figure(figsize=(10, 7))
plt.plot(epoch_count, train_loss_R, 'b-', color='orange')
plt.plot(epoch_count, val_Acc, 'b-', color='green')
plt.plot(epoch_count, train_reward, 'b-', color='blue')
plt.legend(['Training Loss', 'Validation Accuracy', 'Training Reward'])
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
path_plot = os.path.join(ns_vqa_root, 'data/reason', DATA_FOLDER_NAME,'reinforce-plot.png')
plt.savefig(path_plot)
plt.show()
    


# In[ ]:


#PREPROCESS TEST

run = f"python ./reason/tools/preprocess_questions.py         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}         --data_folder_name={DATA_FOLDER_NAME}         --vocab_flag=2" 
get_ipython().system('{run} ')


# In[ ]:


#TESTING
run = f"python ./reason/tools/run_test.py         --main_root={main_root}         --ns_vqa_root={ns_vqa_root}         --data_folder_name={DATA_FOLDER_NAME}         --clevr_val_scene_path {main_root}/clevr-poc-dataset-gen/{DATA_FOLDER_NAME}/incomplete/scenes/testing         --clevr_val_question_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-test_questions.h5         --clevr_constraint_scene_path {main_root}/clevr-poc-dataset-gen/environment_constraints         --load_checkpoint_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/model_reinforce/checkpoint_best.pt         --save_result_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/test_result.json         --clevr_vocab_path {ns_vqa_root}/data/reason/{DATA_FOLDER_NAME}/clevr-poc-vocab.json"
get_ipython().system('{run} ')


# 
