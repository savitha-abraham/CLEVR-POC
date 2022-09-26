#!/bin/bash 

#!/bin/bash


use_gpu=0
render_batch_size=2 
start_question_idx=0

dataset_names=('training' 'testing' 'validation')
dataset_sizes=(100 10 10)


num_images=$((${dataset_sizes[0]}/10))

cd image_generation


length=${#dataset_names[@]}
for (( i = 0; i < length; i++ )); do
	blender --background -noaudio --python render_images.py -- --num_images $num_images --split ${dataset_names[i]} --use_gpu  $use_gpu --render_batch_size $render_batch_size --start_question_idx $start_question_idx

	cd ../question_generation
   
   	python generate_questions.py --split ${dataset_names[i]}
        
   	cd ../image_generation
        
   	blender --background -noaudio --python render_incomplete_images.py -- --split ${dataset_names[i]} --use_gpu  $use_gpu --render_batch_size $render_batch_size --start_question_idx $start_question_idx
       echo "--------------------------------------------------------------------"
done

##cd ../question_generation
##python generate_questions.py
##cd ../image_generation
