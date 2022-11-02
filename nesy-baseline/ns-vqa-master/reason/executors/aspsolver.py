import os
import json

def getInPredicate(scene_dict):
    preds = []
    relations = scene_dict["relationships"]
    
    for key in relations:
      list_obj = relations[key]
      for l in range(len(list_obj)):
          if len(list_obj[l]) == 0:
            continue
          for obj in list_obj[l]:
              pred = key+"("+str(l)+","+str(obj)+")."
              pred = pred.replace('"','')
              pred = pred.strip('\"')
              pred = pred.strip('\'')
              preds.append(pred)
  
    objects =  scene_dict["objects"]
   
    for i in range(len(objects)):
      obj_dict = objects[i]
      shape = "hasProperty("+str(i)+",shape,"+obj_dict["shape"]+")."
      shape = shape.replace('"','')
      shape = shape.strip('\"')
      preds.append(shape)
      
      size = "hasProperty("+str(i)+",size,"+obj_dict["size"]+")."
      size = size.replace('"','')
      size = size.strip('\"')
      preds.append(size)
      
      color = "hasProperty("+str(i)+",color,"+obj_dict["color"]+")."
      color = color.replace('"','')
      color = color.strip('\"')
      preds.append(color)
      
      material = "hasProperty("+str(i)+",material,"+obj_dict["material"]+")."
      material = material.replace('"','')
      material = material.strip('\"')
      preds.append(material)

      region = "at("+str(i)+","+obj_dict["region"]+")."
      region = region.replace('"','')
      region = region.strip('\"')
      preds.append(region)
    return preds

def getToken_program(seq_ids, idx_to_token):
      tokens = ""
      for i in seq_ids:
        if (i.item()==0 or i.item()==1 or i.item()==2 or i.item()==3):
          continue  
        tokens= tokens+idx_to_token[i.item()]+','
      tokens = 'missing(Q):-'+tokens[:-1]+'.'
      return tokens


def getToken(seq_ids, idx_to_token):
      tokens = ""
      for i in seq_ids:
        if (i.item()==0 or i.item()==1 or i.item()==2 or i.item()==3):
          continue  
        tokens= tokens+" "+idx_to_token[i.item()]
      return tokens


def solve(pred_pgm, scene_filename,  constraint_type_index, split, scene_folder, env_folder):
    if pred_pgm == "":
        return None
    name = str(scene_filename.tolist())
    
    num_zero = 6-len(name)
    zeros = ""
    for z in range(0, num_zero):
      zeros = zeros+"0"
    scene_filename = "CLEVR_"+zeros+name+".json"
    
    scene_file_path = os.path.join(scene_folder, scene_filename)
    with open(scene_file_path, encoding="utf-8") as f:
        scene_dict = json.load(f)             
    
    complete = ""
    
    #Read from asp_theory file of constraint_type_index
    constraint_path = os.path.join(env_folder, str(constraint_type_index)+'.lp')
    file2 = open(constraint_path, 'r')
    Lines = file2.readlines()
    complete = ""
    for line in Lines:
      if "#show" in line:
        continue
      complete = complete+line
    file2.close()
    
    #Add definitions for same_color, same_material, same_size, same_shape
    complete = complete+'\n'+':- not missing(_).'+'\n'
    complete = complete+'\n'+'same_color(X,Y):- sameProperty(X, Y, color).'
    complete = complete+'\n'+'same_size(X,Y):- sameProperty(X, Y, size).'
    complete = complete+'\n'+'same_shape(X,Y):- sameProperty(X, Y, shape).'
    complete = complete+'\n'+'same_material(X,Y):- sameProperty(X, Y, material).'
	
    #Add scene information
    
    scene_predicates = getInPredicate(scene_dict)
    for pred in scene_predicates:
      complete = complete+pred+"\n"
    complete = complete+pred_pgm
    complete = complete+"\n"+"#show missing/1."
    
    temp_file = "temp.lp"
    file1 = open(temp_file, 'w')
    n1 = file1.write(complete)
    file1.close()
    if constraint_type_index==122:
    	print('COMPLETE:::', complete)
		
    asp_command = 'clingo 0'  + ' ' + temp_file
    output_stream = os.popen(asp_command)
    output = output_stream.read()
    #print('OUTPUT::')
    #print(output)
    possible_values = None
    if ("Answer" in output):
        answers = output.split('Answer:')
        #print("Answers:", answers)
        answers = answers[1:]
        possible_values = []
        for answer_index, answer in enumerate(answers):
            ans = answer.split('\n')[1].split(' ')
            val = ans[0][8:(len(ans[0])-1)]
            if val=='':
                continue
            if (val not in possible_values):
                possible_values.append(val)
    temp_path = os.path.join(temp_file)
    if os.path.isfile(temp_path):
        os.remove(temp_path)
        
    return possible_values    
   

    
