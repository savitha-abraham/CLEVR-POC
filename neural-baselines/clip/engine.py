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

            data_device = to_device(data, device)

            images = list(map(get_pil_image, data_device['image_path']))

            inputs = clip_processor(text=data_device['question'], images=images, return_tensors="pt", padding=True)
        
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




def test(model, dataloader, device):
    for counter, data in enumerate(dataloader):
        image, target = data['image'].to(device), data['label']
        # get all the index positions where value == 1
        target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
        # get the predictions by passing the image through the model
        outputs = model(image)
        outputs = torch.sigmoid(outputs)
        outputs = outputs.detach().cpu()
        sorted_indices = np.argsort(outputs[0])
        best = sorted_indices[-3:]
        string_predicted = ''
        string_actual = ''
        for i in range(len(best)):
            string_predicted += f"{genres[best[i]]}    "
        for i in range(len(target_indices)):
            string_actual += f"{genres[target_indices[i]]}    "
        image = image.squeeze(0)
        image = image.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
        plt.savefig(f"../outputs/inference_{counter}.jpg")
        plt.show()