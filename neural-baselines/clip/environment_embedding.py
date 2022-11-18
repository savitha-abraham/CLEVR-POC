import torch
import os
import pickle
from transformers import AutoTokenizer, AutoModel

def get_file(env_folder, constraint_type_index):
    sentences = []

    file1 = open(os.path.join(env_folder, str(constraint_type_index) + '.lp'), 'r')
    # Using for loop
    for line in file1:
        sentences.append(line.strip())
    return sentences

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_environment_embedding(env_folder, constraint_type_index, tokenizer, model):

    sentences = get_file(env_folder, constraint_type_index)   
    tokenizer.pad_token = tokenizer.eos_token


    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding = True, truncation=True, return_tensors='pt')
    #print(torch.size(encoded_input))
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])


    # Perform pooling. In this case, mean pooling.
    lp_embeddings = pooled = torch.mean(sentence_embeddings, 0)#, encoded_input['attention_mask'])


    #from sentence_transformers import SentenceTransformer
    #sentences = [data]

    #model = SentenceTransformer('{MODEL_NAME}')
    #embeddings = model.encode(sentences)
    #print(embeddings)
    return lp_embeddings


def get_total_embedding(env_folder):
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')    
    gpt2_model = AutoModel.from_pretrained('gpt2')

    total_embedding = {}
    all_file_names = [f for f in os.listdir(env_folder) if '.lp' in f]
    for f in all_file_names:
        print(f)         
        constraint_type_index = int(f.split('.')[0])
        total_embedding[constraint_type_index] = get_environment_embedding(env_folder, constraint_type_index, gpt2_tokenizer, gpt2_model)         
    
    with open(os.path.join(env_folder, 'total_embedding.pickle'), 'wb') as t:
        pickle.dump(total_embedding, t) 


if __name__ == "__main__":
    env_folder = '/home/marjan/code/CLEVR-POC/clevr-poc-dataset-gen/environment_constraints'
    get_total_embedding(env_folder)