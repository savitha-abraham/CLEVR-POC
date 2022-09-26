import json
import scene_info

with open('../data/constraints.json', 'r') as f_constraints:
    constraint_types = json.load(f_constraints)


with open('../data/properties.json', 'r') as f:
    properties = json.load(f)

dict = {}

for key, value in constraint_types.items():
    c_id = key.split('_')[2]
    regions = value['regions']
    
    for region_id, region in enumerate(regions):
        r = scene_info.Region(constraints=region['constraints'], properties=properties)
        solutions = r.get_all_solutions()

        l = []    
        for s in solutions:
            s = [s[f] for f in sorted(s.keys())]
            s.append(region_id)
            l.append(s)
        dict[(int(c_id), region_id)]  = l
        
print(dict)


import pickle
pickle.dump(dict, open( "consistent_combination.p", "wb" ) )
        

