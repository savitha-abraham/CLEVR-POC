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
              pred = key+"("+str(l)+","+obj+")."
              pred = pred.replace('"','')
              pred = pred.strip('\"')
              preds.append(pred)
    
    objects =  scene_dict["objects"] 
    for i in range(len(objects)):
      obj_dict = objects[i]

      shape = "hasProperty("+str(i)+",shape,"+obj_dict["shape"]+")."
      shape = shape.replace('"','')
      shape = shape.strip('\"')
      preds.append(pred)
      
      size = "hasProperty("+str(i)+",size,"+obj_dict["size"]+")."
      size = size.replace('"','')
      size = size.strip('\"')
      preds.append(pred)
      
      color = "hasProperty("+str(i)+",color,"+obj_dict["color"]+")."
      color = color.replace('"','')
      color = color.strip('\"')
      preds.append(pred)
      
      material = "hasProperty("+str(i)+",material,"+obj_dict["material"]+")."
      material = material.replace('"','')
      material = material.strip('\"')
      preds.append(pred)

      region = "at("+str(i)+","+obj_dict["region"]+")."
      region = region.replace('"','')
      region = region.strip('\"')
      preds.append(pred)

    return preds



def solve(pred_pgm, scene_filename,  constraint_type_index, split, scene_folder, env_folder):
    scene_file_path = os.path.join(scene_folder, scene_filename)
    with open(scene_file_path, encoding="utf-8") as f:
        scene_dict = json.load(f)             
    
    complete = ""
    
    #Read from asp_theory file of constraint_type_index
    #constraint_path = os.path.join(env_folder, str(constraint_type_index)+'.lp')
    #file2 = open(constraint_path, 'r')
    #Lines = file2.readlines()
    #complete = ""
    #for line in Lines:
    #  if "#" in line:
    #    continue
    #  complete = complete+line
    #file2.close()

    #Add scene information
    
    scene_predicates = getInPredicate(scene_dict)
    print(scene_predicates)
    for pred in scene_predicates:
      complete = complete+pred+"\n"
    complete = complete+pred_pgm
    complete = complete+"\n"+"#show missing/1."
    temp_file = "temp.lp"
    file1 = open(temp_file, 'w')
    n1 = file1.write(complete)
    file1.close()

    asp_command = 'clingo 0'  + ' ' + temp_file
    output_stream = os.popen(asp_command)
    output = output_stream.read()
    answers = output.split('Answer:')
    #print("Answers:", answers)
    answers = answers[1:]
    possible_values = []
    for answer_index, answer in enumerate(answers):
        ans = answer.split('\n')[1].split(' ')
        val = ans[0][8:(len(ans[0])-1)]
        if (val not in possible_values):
            possible_values.append(val)
    temp_path = os.path.join(temp_file)
    if os.path.isfile(temp_path):
        os.remove(temp_path)
    if len(possible_values) == len(domain[query_attribute]):
        return None
    elif len(possible_values) < len(domain[query_attribute]):
        return possible_values 


    
