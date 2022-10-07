import os, math
## sizes are w.r.t incomplete dataset i.e., number of incomplete scenes
training_size = 4000 #900000
testing_size = training_size/10
validation_size = training_size/10
num_constraints_per_round = 2

use_gpu=0
render_batch_size=40
start_idx=0

dataset_names=['training', 'testing', 'validation']
dataset_sizes=[training_size, testing_size, validation_size]
total_number_env_training = (int(math.ceil(dataset_sizes[0]/render_batch_size)*num_constraints_per_round))

num_constraint_types = [num_constraints_per_round, total_number_env_training, total_number_env_training]


os.chdir('image_generation')
os.system('echo $PWD')
for i, dataset in enumerate(dataset_names):
    
    
    num_images = int(math.ceil(dataset_sizes[i]))
    input(num_images)
    start_idx=0
    while(True):
        os.system('blender --background -noaudio --python render_images.py -- --num_images ' + str(num_images) + ' --split ' + dataset_names[i] + ' --use_gpu ' + str(use_gpu) + ' --render_batch_size ' + str(render_batch_size) + ' --start_idx ' + str(start_idx)) + ' --num_constraint_types ' + str(num_constraint_types[i])
        start_idx += render_batch_size
        if start_idx >= num_images:
            break
        print('complete: start_index_', start_idx)
    
    