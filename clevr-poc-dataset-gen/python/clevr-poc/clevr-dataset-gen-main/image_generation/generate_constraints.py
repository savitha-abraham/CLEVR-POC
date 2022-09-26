import json
import os
import numpy
import collections 
import copy


def main():
<<<<<<< HEAD
    """
=======
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109
    x = {
        "regions": [
            {# region 0
                    "range": {"x":[-3.5, -1.17], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [0, 1],
                        "material":[1],
                        "color_shape": [[0, 0], [1, 2], [1, 1]],
                        "shape_size": [[1, 1], [2, 0], [0, 0]],
                    }
            },
            {# region 1
                    "range": {"x":[-1.16, 1.17], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [1, 2],
                        "material":[0],
                        "size":[1],
                        "color_shape": [[1, 0], [2, 1]]
                    }
            },
            {# region 2
                    "range": {"x":[1.18, 3.5], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [2, 3],
                        "shape":[2],
                        "size":[1],
                        "color_material": [[2, 0], [3, 1]]                   
                    }
            },
            {# region 3
                    "range": {"x":[-3.5, -1.17], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [3, 4],
                        "material":[0],
                        "shape":[0, 1],
                        "size":[1],
                        "color_shape": [[3, 0], [4, 1]]      
                    }
            },
            {# region 4
                    "range": {"x":[-1.16, 1.17], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [4, 5],
                        "material":[0, 1],
                        "shape":[2],
                        "size":[0, 1],
                        "material_size": [[1, 0], [0, 1]],
                        "color_size": [[4, 0], [5, 1]], 
                    }
            },
            {# region 5
                    "range": {"x":[1.18, 3.5], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [5, 6],
                        "material":[0],
                        "shape":[0, 1],
                        "shape_size": [[1, 1], [0, 0]],
                        "color_shape": [[5, 1], [6, 0]],
                    }
            },
            {# region 6
                    "range": {"x":[-3.5, -1.17], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [6, 7],
                        "material":[1],
                        "shape":[0, 2],
                        "size":[0],
                        "color_shape": [[6, 2], [7, 0]],
                    }
            },
            {# region 7
                    "range": {"x":[-1.16, 1.17], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [7, 0],
                        "material":[0],
                        "shape":[0],
                        "color_size": [[7, 1], [0, 0]],
                    }
            },
            {# region 8
                    "range": {"x":[1.18, 3.5], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [0, 1],
                        "shape":[1],
                        "size":[0],
                        "color_material": [[0, 1], [1, 0]],
                    }
            }
        ]
    }
<<<<<<< HEAD
    """

    x = {
        "regions": [
            {# region 0
                    "range": {"x":[-3.5, -1.17], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [0],
                        "material":[1],
                        "shape": [0],
                        "size": [0]
                    }
            },
            {# region 1
                    "range": {"x":[-1.16, 1.17], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [1],
                        "material":[0],
                        "shape":[1],
                        "size":[0]
                    }
            },
            {# region 2
                    "range": {"x":[1.18, 3.5], "y":[1.18, 3.5]},
                    "constraints":{
                        "color": [2],
                        "material":[0],
                        "shape":[2],
                        "size":[1]
                    }
            },
            {# region 3
                    "range": {"x":[-3.5, -1.17], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [3],
                        "material":[0],
                        "shape":[0],
                        "size":[1]
                    }
            },
            {# region 4
                    "range": {"x":[-1.16, 1.17], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [4],
                        "material":[0],
                        "shape":[2],
                        "size":[1]
                    }
            },
            {# region 5
                    "range": {"x":[1.18, 3.5], "y":[-1.16, 1.17]},
                    "constraints":{
                        "color": [5],
                        "material":[0],
                        "shape":[2],
                        "size": [0]
                    }
            },
            {# region 6
                    "range": {"x":[-3.5, -1.17], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [6],
                        "material":[1],
                        "shape":[1],
                        "size":[1]
                    }
            },
            {# region 7
                    "range": {"x":[-1.16, 1.17], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [7],
                        "material":[0],
                        "shape":[0],
                        "size": [0]
                    }
            },
            {# region 8
                    "range": {"x":[1.18, 3.5], "y":[-3.5, -1.17]},
                    "constraints":{
                        "color": [8],
                        "material":[1],
                        "shape":[2],
                        "size":[0]
                    }
            }
        ]
    }
=======
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109


    output_dir = os.path.join(os.path.dirname(__file__), '../data/properties.json')   
    with open(output_dir, 'r') as f_properties:
        properties = json.load(f_properties)
    

    sorted_properties = {
        'shape': get_sorted_list(properties, 'shape'),
        'material': get_sorted_list(properties, 'material'),
        'size': get_sorted_list(properties, 'size'),
        'color': get_sorted_list(properties, 'color')
    }


    regions = x["regions"]
    for region in regions:
        constraints = {}
        for key, value in region['constraints'].items():
            key_names = key.split('_')
            if len(key_names) == 1:
                list = sorted_properties[key_names[0]]    
                constraints[key] = [list[v] for v in value]
            else: # len = 2
                list1 = sorted_properties[key_names[0]]    
                list2 = sorted_properties[key_names[1]]    
                constraints[key] = [[list1[v[0]], list2[v[1]]] for v in value]

        region['constraints'] = constraints

    
    x['regions'] = regions
    rolling_regions = copy.deepcopy(regions)
    
    constraints = {}
    for i in range(len(rolling_regions)):
        
        rolling_regions = numpy.roll(rolling_regions, i).tolist()

        for region_index in range(len(rolling_regions)):
            rolling_regions[region_index]['range'] = regions[region_index]['range']

        
        regions_info = {}
        regions_info["regions"] = rolling_regions
        

        key_name = "constraint_type_" + str(i)
        constraints[key_name] = copy.deepcopy(regions_info)
        
        """
        print(constraints[key_name]['regions'][1]['range'], key_name)
        print(regions_info['regions'][1]['range'])
        print(rolling_regions[1]['range'])
        print(regions[1]['range'])
        input('---------------')
        """
    
    
    """
    print(constraints["constraint_type_0"]['regions'][1]['range'])
    print(constraints["constraint_type_1"]['regions'][1]['range'])
    print(constraints["constraint_type_2"]['regions'][1]['range'])
    print(regions[1]['range'])
    print('-----------------')
    """


    
    with open(os.path.join(os.path.dirname(__file__), '../data/constraints.json'), 'w') as output_file:
        json.dump(constraints, output_file, sort_keys = True, indent=3)
    print("The file constraints.json is created/updated!")



<<<<<<< HEAD

=======
>>>>>>> 1420a7e8e6a6f8c8266ec549e5cce8d30e259109
def get_sorted_list(dict, property_name):
    l = dict[property_name]
    return sorted([*l])
    

if __name__ == '__main__':
    main()