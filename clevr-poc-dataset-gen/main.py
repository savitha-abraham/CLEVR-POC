import os, math
## sizes are w.r.t incomplete dataset i.e., number of incomplete scenes
training_size = 1000 #900000
testing_size = training_size/10
validation_size = training_size/10


use_gpu=0
render_batch_size=100
start_idx=0

dataset_names=['training', 'testing', 'validation']
dataset_sizes=[training_size, testing_size, validation_size]

os.chdir('image_generation')
os.system('echo $PWD')
for i, dataset in enumerate(dataset_names):
    
    
    num_images = int(math.ceil(dataset_sizes[i]/10))
    input(num_images)
    start_idx=0
    while(True):
        os.system('blender --background -noaudio --python render_images.py -- --num_images ' + str(num_images) + ' --split ' + dataset_names[i] + ' --use_gpu ' + str(use_gpu) + ' --render_batch_size ' + str(render_batch_size) + ' --start_idx ' + str(start_idx))
        start_idx += render_batch_size
        if start_idx >= num_images:
            break
        print('complete: start_index_', start_idx)
    
    os.chdir('../question_generation')
    os.system('python generate_questions.py --split ' + dataset_names[i])
    os.chdir('../image_generation')

    """
    num_images = int(math.ceil(dataset_sizes[i]))
    start_idx = 0
    while(True):
        os.system('blender --background -noaudio --python render_incomplete_images.py -- --split ' + dataset_names[i] + ' --use_gpu ' + str(use_gpu) + ' --render_batch_size ' + str(render_batch_size) + ' --start_question_idx ' + str(start_idx))
        start_idx += render_batch_size
        if start_idx >= num_images:
            break
        print('incomplete: start_question_index_', start_idx)
    """
