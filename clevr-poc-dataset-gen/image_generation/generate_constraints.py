import os, json, random


ASP_FILE_PATH = '/home/marjan/myworks/code/asp/constraints.lp'
MAX_NUMBER_OF_ANSWERS = 1000000
TOTAL_NUMBER_OF_CONSTRAINT_TYPES = 30
ranges = [
    {"x":[-3.5, -1.17], "y":[1.18, 3.5]},
    {"x":[-1.16, 1.17], "y":[1.18, 3.5]},
    {"x":[1.18, 3.5], "y":[1.18, 3.5]},
    {"x":[-3.5, -1.17], "y":[-1.16, 1.17]},
    {"x":[-1.16, 1.17], "y":[-1.16, 1.17]},
    {"x":[1.18, 3.5], "y":[-1.16, 1.17]},
    {"x":[-3.5, -1.17], "y":[-3.5, -1.17]},
    {"x":[-1.16, 1.17], "y":[-3.5, -1.17]},
    {"x":[1.18, 3.5], "y":[-3.5, -1.17]}
]




## run asp
asp_command = 'clingo ' + str(MAX_NUMBER_OF_ANSWERS) + ' ' + ASP_FILE_PATH
output_stream = os.popen(asp_command)
output = output_stream.read()

## parsing answer sets
x = {}
answers = output.split('Answer:')
key_prefix = 'constraint_type_'

answers = answers[1:]
random.shuffle(answers)

random_answers = random.sample(answers, k=TOTAL_NUMBER_OF_CONSTRAINT_TYPES)


for answer_index, answer in enumerate(random_answers):

    constraint_type_index = key_prefix + str(answer_index)
    regions_constraints = [[] for i in range(len(ranges))]
    
    constraints = answer.split('\n')[1].split(' ')
    for constraint in constraints:
        features = constraint.split('(')[1].split(')')[0].split(',')
        region_index = int(features[0])
        regions_constraints[region_index].append({'color': features[1], 'material': features[2], 'shape': features[3], 'size': features[4]})

    regions_info = []
    for region_index, region_constraints in enumerate(regions_constraints):
        info = {}
        info['constraints'] = region_constraints
        info['range'] = ranges[region_index]
        regions_info.append(info)

    #input(regions_info)
    x[constraint_type_index] = {
        "regions": regions_info
    }

## generating constraint_types dictionary



with open(os.path.join(os.path.dirname(__file__), '../data/constraints.json'), 'w') as output_file:
    json.dump(x, output_file, sort_keys = True, indent=3)
print("The file constraints.json is created/updated!")