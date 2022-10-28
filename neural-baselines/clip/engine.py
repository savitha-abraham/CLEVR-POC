import torch
from tqdm import tqdm
import torch.nn as nn
from PIL import Image

def to_device(data, device):
        
        constraint_embedding = data['constraint_embedding'].to(device)
        target = data['target'].to(device)
        
        #input_ids = data['input_ids'].to(device)   
        #pixel_values = data['pixel_values'].to(device)   
        #attention_mask = data['attention_mask'].to(device)   

        return {
            'constraint_embedding': constraint_embedding, 
            'target': target,

            #'input_ids': input_ids,
            #'pixel_values': data['pixel_values'],
            #'attention_mask': data['attention_mask'],
            
            'image_path': data['image_path'],
            #'clip_input': data['clip_input'],
            'question': data['question'],
            'constraint_type': data['constraint_type'],
            'answer': data['answer']
        }

def get_pil_image(image_path):
    return Image.open(image_path).convert("RGB")  

def train(final_classifier, clip_model, dataloader, optimizer, criterion, train_data, device, dropout, clip_processor):
    print('Training')
    final_classifier.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        counter += 1
        
        data_device = to_device(data, device)
       

        images = list(map(get_pil_image, data_device['image_path']))

        inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        clip_output = clip_model(**inputs)
        
        text_emb = clip_output['text_embeds']
        image_emb = clip_output['image_embeds']
        
        constraint_type_embedding = data_device['constraint_embedding']

        emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
        emb = dropout(emb)


        optimizer.zero_grad()
        
        outputs = final_classifier(emb)

        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        
        loss = criterion(outputs, data_device['target'])
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()


        
    train_loss = train_running_loss / counter
    return train_loss




# validation function
def validate(final_classifier, clip_model, dataloader, criterion, val_data, device, dropout, clip_processor):
    print('Validating')
    final_classifier.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            counter += 1
            
            data_device = to_device(data, device)

            images = list(map(get_pil_image, data_device['image_path']))

            inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            clip_output = clip_model(**inputs)
            
            text_emb = clip_output['text_embeds']
            image_emb = clip_output['image_embeds']
            constraint_type_embedding = data_device['constraint_embedding']

            emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
            emb = dropout(emb)
            
            outputs = final_classifier(emb)

            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
        
            loss = criterion(outputs, data_device['target'])
            val_running_loss += loss.item()
        
        val_loss = val_running_loss / counter
        return val_loss       



def test(final_classifier, clip_model, dataloader, criterion, test_data, device, dropout, clip_processor):
    print('Testing')
    final_classifier.eval()
    counter = 0
    correct = 0
    
    for i, data in tqdm(enumerate(dataloader), total=int(len(test_data)/dataloader.batch_size)):
        counter += 1
        
        data_device = to_device(data, device)
        images = list(map(get_pil_image, data_device['image_path']))
       
        
        inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
        inputs = inputs.to(device)
        clip_output = clip_model(**inputs)
            
        text_emb = clip_output['text_embeds']
        image_emb = clip_output['image_embeds']
        constraint_type_embedding = data_device['constraint_embedding']

        emb = torch.cat([text_emb, image_emb, constraint_type_embedding], dim=1)          
        emb = dropout(emb)
          
        outputs = final_classifier(emb)

        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)        
        
        #loss = criterion(outputs, data_device['target'])
        #test_running_loss += loss.item()

        pred = torch.max(outputs, 1)[1]
        t = torch.max(data_device['target'], 1)[1]
        correct += pred.eq(t).sum()
        
        

        """
        ## assign 0 label to those with less than 0.5
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0 

        N,C = data_device['target'].shape
        accuracy = (outputs == data_device['target']).sum() / (N*C) * 100   
        """
    #test_loss = test_running_loss / counter
    accuracy = 100 * correct / counter
    return accuracy
