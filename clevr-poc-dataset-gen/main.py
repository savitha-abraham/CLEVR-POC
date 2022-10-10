import os, math
## sizes are w.r.t incomplete dataset i.e., number of incomplete scenes
num_constraint_types = 2
training_size = 6 #900000
testing_size = int(math.ceil(training_size/10))
validation_size = int(math.ceil(training_size/10))
num_constraints_per_round = 2

use_gpu=0
render_batch_size=40
start_idx=0

dataset_names=['training', 'testing', 'validation']
dataset_sizes=[training_size, testing_size, validation_size]



os.chdir('image_generation')
os.system('echo $PWD')
for i, dataset in enumerate(dataset_names):
    
    num_images = int(math.ceil(dataset_sizes[i]))
    start_idx=0
    while(True):
        os.system('blender --background -noaudio --python render_images.py -- --num_images ' + str(num_images) + ' --split ' + dataset_names[i] + ' --use_gpu ' + str(use_gpu) + ' --render_batch_size ' + str(render_batch_size) + ' --start_idx ' + str(start_idx) + ' --num_constraint_types ' + str(num_constraint_types) + ' --phase_constraint 0')
        start_idx += render_batch_size
        if start_idx >= num_images:
            break
        print('complete: start_index_', start_idx)
    
    