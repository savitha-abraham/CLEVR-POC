import sys, os, argparse, json
from pathlib import Path

path_root = os.path.dirname(os.getcwd())
sys.path.append(str(path_root))

from generate_dataset import parser
from image_generation import scene_info

def get_relations_in_scene(relations, direction, object_indx, new_relations, start_indx):
    n = len(relations[direction][object_indx])
    for i in range(n):
        new_relations[str(start_indx + i)] = {'name': direction, 'object':relations[direction][object_indx][i]}

    start_indx = start_indx + n
    return start_indx, new_relations


def represent_relations(relations, relation_id, name, region_destination_ids):
    for destination_id in region_destination_ids:
        relations[relation_id] = {'name': name, 'region': destination_id}
        relation_id += 1
    return relations, relation_id

def get_region_relations(region_index):
    relations = {}
    relation_id = 0
    if region_index == 0:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2])
    elif region_index == 1:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6, 1, 4, 7])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[1, 4, 7, 2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2])        
    elif region_index == 2:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2])                    
 
    elif region_index == 3:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[3, 4, 5, 6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2, 3, 4, 5])
    elif region_index == 4:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6, 1, 4, 7])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[1, 4, 7, 2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[3, 4, 5, 6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2, 3, 4, 5])          
    elif region_index == 5:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[3, 4, 5, 6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=[0, 1, 2, 3, 4, 5])
    elif region_index == 6:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=list(range(9)))    
    elif region_index == 7:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=[0, 3, 6, 1, 4, 7])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[1, 4, 7, 2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=list(range(9)))    

    elif region_index == 8:
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='left', region_destination_ids=list(range(9)))
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='right', region_destination_ids=[2, 5, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='front', region_destination_ids=[6, 7, 8])
        relations, relation_id = represent_relations(relations=relations, relation_id=relation_id, name='behind', region_destination_ids=list(range(9)))   


    return relations

def reconstruct_scene(args, scene_dir, constraint_types, properties):
    sorted_key_properties = sorted(properties.keys())

    #a dictionary to represent scenes compatible for the neural network baselines
    new_scenes = {}

    with open(os.path.join(scene_dir, args.split + '.json'), 'r') as f:
        scenes = json.load(f)['scenes']
    

    for scene_indx, scene in enumerate(scenes):
        
        cobstraints = {}

        #extracting scene constraints info
        scene_constraints = constraint_types['_'.join(['constraint_type', str(scene['constraint_type'])])]['regions']

        for region_index, region_constraint in enumerate(scene_constraints):
            
            region = scene_info.Region(constraints=region_constraint['constraints'], properties=properties)
            solutions = region.get_all_solutions()

            attributes = []
            
            for solution in solutions:
                attributes.append([solution[p] for p in sorted_key_properties])
            
            cobstraints[region_index] = {
                'name': '_'.join(['region_constraint', str(region_index)]),
                'attributes': attributes,
                'relations': get_region_relations(region_index=region_index)
            }
            
        objects = {}
        for object_indx, object in enumerate(scene['objects']):
            start_indx = 0
            new_relations = {}
            start_indx, new_relations = get_relations_in_scene(relations=scene['relationships'], direction='left', object_indx=object_indx, new_relations=new_relations, start_indx=start_indx)
            start_indx, new_relations = get_relations_in_scene(relations=scene['relationships'], direction='right', object_indx=object_indx, new_relations=new_relations, start_indx=start_indx)
            start_indx, new_relations = get_relations_in_scene(relations=scene['relationships'], direction='behind', object_indx=object_indx, new_relations=new_relations, start_indx=start_indx)
            start_indx, new_relations = get_relations_in_scene(relations=scene['relationships'], direction='front', object_indx=object_indx, new_relations=new_relations, start_indx=start_indx)

            objects[str(object_indx)] = {
                'name': '_'.join(['obj', str(object_indx)]),

                'x': object['pixel_coords'][0],
                'y': object['pixel_coords'][1],
                'z': object['pixel_coords'][2],
                'attributes': [object[p] for p in sorted_key_properties],
                'relations': new_relations
            }

        new_scenes[str(scene_indx)] = {
            'width': args.width,
            'height': args.height,
            'content': {'objects': objects, 'constraints': cobstraints}
        }

        
    with open(os.path.join(scene_dir, args.split + '_nn.json'), 'w') as f:
        scene = json.dump(new_scenes, f)

def main(args):
    with open(args.properties_json, 'r') as f:
        properties = json.load(f)

    with open(args.constraints_json, 'r') as f_constraints:
        constraint_types = json.load(f_constraints)

    #complete scenes
    scene_dir = os.path.join(args.complete_data_dir, args.scene_dir)
    reconstruct_scene(args, scene_dir=scene_dir, constraint_types=constraint_types, properties=properties)

    #incomplete scenes
    scene_dir = os.path.join(args.incomplete_data_dir, args.scene_dir)
    reconstruct_scene(args, scene_dir=scene_dir,constraint_types=constraint_types, properties=properties)
    


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
