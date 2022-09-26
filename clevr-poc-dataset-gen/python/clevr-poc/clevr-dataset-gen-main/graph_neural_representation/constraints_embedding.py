<<<<<<< HEAD
from copyreg import pickle
=======
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109
from curses import keyname
from ntpath import join
import sys, os, argparse, json, collections
from pathlib import Path
from turtle import shape
import numpy as np
<<<<<<< HEAD
import torch
import pickle

=======
from torch import embedding
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109

path_root = os.path.dirname(os.getcwd())
sys.path.append(str(path_root))

from image_generation import scene_info
from generate_dataset import parser  



def get_dict_all_property_combinations(properties):
    
    sorted_key_properties = sorted(properties.keys())
    key_properties_values = []
    for key_property in sorted_key_properties:
        key_properties_values.append(sorted(properties[key_property].keys()))
    return  {'-'.join([kp0, kp1, kp2, kp3]): 0 for kp0 in key_properties_values[0] for kp1 in key_properties_values[1] for kp2 in key_properties_values[2] for kp3 in key_properties_values[3]}

    

def get_constraint_embedding(dict_all_property_combinations, properties, constraints):
    sorted_key_properties = sorted(properties.keys())
    region = scene_info.Region(constraints=constraints, properties=properties)
    solutions = region.get_all_solutions()
    for solution in solutions:

        #key_name = '-'.join([solution['color'], solution['material'], solution['shape'], solution['size']])
        key_name = '-'.join([solution[p] for p in sorted_key_properties])
<<<<<<< HEAD
=======
        print(key_name)
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109
        dict_all_property_combinations[key_name] = 1
    
    embedding = [dict_all_property_combinations[f] for f in sorted(dict_all_property_combinations.keys())]
    return np.array([embedding])
    
<<<<<<< HEAD
    
=======
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109



    #sorted_properties = collections.OrderedDict(sorted(properties.items()))
    #properties = list(sorted_properties.items())



def main(args):
    

    with open(args.properties_json, 'r') as f:
        properties = json.load(f)
    
    with open(args.constraints_json, 'r') as f_constraints:
        constraint_types = json.load(f_constraints)

<<<<<<< HEAD
    dict_all_property_combinations = get_dict_all_property_combinations(properties=properties)

    constraint_types_tensor = dict.fromkeys(list(range(0, 9))) 
    for constraint_type_index in constraint_types_tensor.keys():
        constraint_type = 'constraint_type_' + str(constraint_type_index)
        scene_constraints = constraint_types[constraint_type]['regions']
        embeddings = np.zeros((len(scene_constraints), len(dict_all_property_combinations)))   #9x96
        for index, region_constraint in enumerate(scene_constraints):
            dict_all_property_combinations = dict.fromkeys(dict_all_property_combinations, 0)
            embeddings[index,:] = get_constraint_embedding(dict_all_property_combinations=dict_all_property_combinations, properties=properties, constraints=region_constraint['constraints'])

        constraint_types_tensor[constraint_type_index] = torch.FloatTensor(embeddings)
    
    with open('constraint_types_tensor.pickle', 'wb') as f:
        pickle.dump(constraint_types_tensor, f)
    
    

if __name__ == '__main__':
    args = parser.parse_args()
   
=======

    #Example
    scene_constraints = constraint_types['constraint_type_0']['regions']

    dict_all_property_combinations = get_dict_all_property_combinations(properties=properties)

    embeddings = np.zeros((len(scene_constraints), len(dict_all_prioperty_combinations)))   #9x96
    for index, region_constraint in enumerate(scene_constraints):
        dict_all_prioperty_combinations = dict.fromkeys(dict_all_property_combinations, 0)
        embeddings[index,:] = get_constraint_embedding(dict_all_property_combinations=dict_all_prioperty_combinations, properties=properties, constraints=region_constraint['constraints'])
    print(embeddings)

if __name__ == '__main__':
    args = parser.parse_args()
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109
    main(args)