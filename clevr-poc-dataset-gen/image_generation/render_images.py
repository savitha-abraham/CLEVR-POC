import math, sys, random, argparse, json, os, tempfile
import collections 
import copy

from pathlib import Path
path_root = Path(__file__).parents[1]
path_current = Path(__file__).parents[0]
sys.path.append(str(path_root))
sys.path.append(str(path_current))



from image_generation import scene_info, blender
from generate_dataset import parser
from generate_environment import generateEnvironment, getSceneGraph
from question_generation.generate_questions import generate_question


INSIDE_BLENDER = True
try:
  import bpy, bpy_extras
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)



def directory_management(main_dir):
  image_dir = os.path.join(main_dir, args.image_dir, args.split)
  scene_dir = os.path.join(main_dir, args.scene_dir)

  
  if not os.path.isdir(image_dir):
    os.makedirs(image_dir) 

  if not os.path.isdir(scene_dir):
    os.makedirs(scene_dir) 

  num_digits = 6

  prefix = '%s_' % (args.filename_prefix)
  image_temp = '%s%%0%dd.png' % (prefix, num_digits)
  scene_temp = '%s%%0%dd.json' % (prefix, num_digits)
  
  scene_template = os.path.join(scene_dir, scene_temp)
  img_template = os.path.join(image_dir, image_temp)



  return scene_dir, img_template, scene_template


##---------------------------------------------------------------------------------------------------------------------------
def main(args):
  

  complete_scene_dir, complete_img_template, complete_scene_template = directory_management(args.complete_data_dir)
  incomplete_scene_dir, incomplete_img_template, incomplete_scene_template = directory_management(args.incomplete_data_dir)


  environment_constraints_dir = os.path.join(args.incomplete_data_dir, args.environment_constraints_dir)
  if not os.path.isdir(environment_constraints_dir):
    os.makedirs(environment_constraints_dir)     


  constraints_types_list = []
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
  
  

  # we have in total 9 regions, and we do not want to have more than one object at each region
  #if args.max_objects > 9:
  #  args.max_objects = 9

  num_image_per_constraint_type = [0 for ind in range(args.num_constraint_types)]
  max_number_of_images_per_constraint = math.floor(args.num_images/args.num_constraint_types)
  
  

  
  #Loading question templates
  templates = {}
  num_loaded_templates = 0
  for fn in os.listdir(os.path.join(str(path_root), "question_generation", args.template_dir)):
      if not fn.endswith('.json'): continue
      with open(os.path.join(str(path_root), "question_generation", args.template_dir, fn), 'r') as f:
          for i, template in enumerate(json.load(f)):
              num_loaded_templates = num_loaded_templates + 1
              key = (fn, i)
              templates[key] = template
          print('Read %d templates from disk' % num_loaded_templates)
  
  num_questions_per_template_type = {}
  for key in templates:
      num_questions_per_template_type[key] = 0
  max_number_of_questions_per_template = math.floor(args.num_images/args.num_templates)
  
  env_id = 0
  objNum_env = {i:[] for i in range(args.min_objects,args.max_objects+1)}
  env_answers = {}
  possible_num_objects = [i for i in range(args.min_objects, args.max_objects)]
  questions = []
  i = args.start_idx
  while i < args.num_images:
  #for i in range(args.num_images):
        possible_sols = None
        complete_scene_graph = {} 
        incomplete_scene_graph = {} 
        query_attribute = "" 
        given_query = "" 
        complete_scene = None
        env_creation_flag = True
        while(possible_sols == None or complete_scene == None):
                #print('HEREEEEE')
                complete_scene_image_path = complete_img_template % i
                incomplete_scene_image_path = incomplete_img_template % i
            

                complete_scene_path = complete_scene_template % i
                incomplete_scene_path = incomplete_scene_template % i
                
                num_objects = random.choice(possible_num_objects)
                
                if (env_id < args.num_constraint_types):
                    generateEnvironment(args, environment_constraints_dir, num_objects, env_id)
                    env_creation_flag = True
                    constraint_type_index = env_id
                else:
                    list_env = objNum_env[num_objects]
                    constraint_type_index = balance_constraint_type(list_env, num_image_per_constraint_type, max_number_of_images_per_constraint)
                    env_creation_flag = False
                    if constraint_type_index == None:
                      possible_num_objects.remove(num_objects)
                      continue
                
                
                
                #Extracting a scene graph conforming to the environment
                #updated_env_ans 
                trials = 0
                while(possible_sols == None and trials<100):
                    complete_scene_graph, incomplete_scene_graph, query_attribute, possible_sols, given_query, obj_rm, updated_env_ans = getSceneGraph(num_objects, constraint_type_index, env_answers, environment_constraints_dir, args)
                    env_answers = copy.deepcopy(updated_env_ans)
                    trials = trials+1
                    
                if possible_sols is not None:
                    print("Scene graph for image ",i, " created!!")
                    complete_scene, incomplete_scene = render_scene(args,
                      complete_scene_graph=complete_scene_graph,
                      incomplete_scene_graph=incomplete_scene_graph,
                      image_index=i,
                      complete_scene_image_path=complete_scene_image_path,
                      incomplete_scene_image_path= incomplete_scene_image_path,
                      properties=properties,
                      constraint_type_index=constraint_type_index
                    )

                    if complete_scene is not None:
                        with open(complete_scene_path, 'w') as f:
                            json.dump(complete_scene, f)  
                        with open(incomplete_scene_path, 'w') as f:
                            json.dump(incomplete_scene, f)
                        if env_creation_flag:
                            objNum_env[num_objects].append(env_id) 
                            num_image_per_constraint_type[env_id]= num_image_per_constraint_type[env_id] +1
                            env_id = env_id +1
                        #Generate question for the scene...
                        question = generate_question(args,templates, num_loaded_templates, query_attribute, given_query, obj_rm, possible_sols, complete_scene, complete_scene_path, i, num_questions_per_template_type, max_number_of_questions_per_template )
                        questions.append(question)
                        #print(question)
                    else:
                        possible_sols = None
                        


        i = i + 1
        if i == args.start_idx + args.render_batch_size:  #to avoid GPU CUDA overflow!
          break

  question_dir = os.path.join(args.incomplete_data_dir, args.question_dir)
  if not os.path.isdir(question_dir):
    os.makedirs(question_dir)

  questions_file = os.path.join(question_dir, args.split + '.json')


  with open(questions_file, 'w') as f:
    print('Writing output to %s' % questions_file)
    json.dump({
        #'info': scene_info,
        'questions': questions,
      }, f)


#-----------------------------------------------------------------------------------------------------------------------


def get_already_rendered_scenes(split, scene_dir):
  if os.path.exists(os.path.join(scene_dir, split + '.json')):
    with open(os.path.join(scene_dir, split + '.json'), 'r') as f:
      data = json.load(f)
    scenes = data['scenes']
  else:
    scenes = []
  
  return scenes


##---------------------------------------------------------------------------------------------------------------------------

def render_scene(args,
      complete_scene_graph=None,
      incomplete_scene_graph=None,
      image_index=0,
      complete_scene_image_path='render.png',
      incomplete_scene_image_path='render.png',
      properties=None,
      constraint_type_index=None

  ):

  blender_obj = blender.Blender(complete_scene_image_path, 
    args.material_dir, 
    args.base_scene_blendfile, 
    args.width, 
    args.height, 
    args.render_tile_size, 
    args.use_gpu,
    args.render_num_samples,
    args.render_min_bounces, 
    args.render_max_bounces)

   

    

  # This will give ground-truth information about the scene and its objects
  complete_scene_struct = {
       'info': {
          'date': args.date,
          'version': args.version,
          'split': args.split,
          'license': args.license,
       },
      'constraint_type_index': constraint_type_index,
      'image_index': image_index,
      'image_filename': os.path.basename(complete_scene_image_path),
      'objects': [],
      'directions': {}
  }


  plane_behind, plane_left, plane_up = blender_obj.get_plane_direction()
  

  # Save all six axis-aligned directions in the scene struct
  complete_scene_struct['directions']['behind'] = tuple(plane_behind)
  complete_scene_struct['directions']['front'] = tuple(-plane_behind)
  complete_scene_struct['directions']['left'] = tuple(plane_left)
  complete_scene_struct['directions']['right'] = tuple(-plane_left)
  complete_scene_struct['directions']['above'] = tuple(plane_up)
  complete_scene_struct['directions']['below'] = tuple(-plane_up)

  


  loop_counter  = 0
  succeed = False
  # Building a (complete) scene and check the validity and visibility of all the randomly added objects
  while (loop_counter < 10):
    objects, objects_blender_info = add_objects(complete_scene_struct, args, properties, complete_scene_graph)
    objects, blender_objects = get_blender_objects(objects, objects_blender_info, blender_obj)
    all_visible = blender_obj.check_visibility(blender_objects, args.min_pixels_per_object)
 
    if not all_visible:
      # If any of the objects are fully occluded then start over; delete all
      # objects from the scene and place them all again.
      print('Some objects are occluded; replacing objects')
      make_scene_empty(blender_objects)        
      loop_counter = loop_counter + 1
    else:
      succeed = True
      break

  if not succeed:
    return None, None
  else:
    
    complete_scene_struct['objects'] = objects
    complete_scene_struct['relationships'] = scene_info.compute_all_relationships(complete_scene_struct)
    complete_scene_struct['similar'] = scene_info.compute_all_similar(complete_scene_struct)
    #scene_struct['objects_blender_info'] = objects_blender_info    
    
    blender_obj.render()

    
    blender_incomplete_obj = blender.Blender(incomplete_scene_image_path, 
      args.material_dir, 
      args.base_scene_blendfile, 
      args.width, 
      args.height, 
      args.render_tile_size, 
      args.use_gpu,
      args.render_num_samples,
      args.render_min_bounces, 
      args.render_max_bounces) 
    
    blender_incomplete_obj.get_plane_direction()

    incomplete_objects, incomplete_blender_info = get_incomplete_scene_info(complete_scene_graph, incomplete_scene_graph, objects, objects_blender_info)
    incomplete_objects, incomplete_blender_objects = get_blender_objects(incomplete_objects, incomplete_blender_info, blender_incomplete_obj)    

    blender_incomplete_obj.render()


    incomplete_scene_struct = copy.deepcopy(complete_scene_struct)
    del incomplete_scene_struct['similar']
    incomplete_scene_struct['image_filename'] = os.path.basename(incomplete_scene_image_path)
    incomplete_scene_struct['objects'] = incomplete_objects
    incomplete_scene_struct['relationships'] = scene_info.compute_all_relationships(incomplete_scene_struct)
     
    

    return complete_scene_struct, incomplete_scene_struct


##---------------------------------------------------------------------------------------------------------------------------

def get_incomplete_scene_info(complete_scene_graph, incomplete_scene_graph, objects, blender_objects):
  obj_interest = None
  for obj in complete_scene_graph:
    if obj not in incomplete_scene_graph:
      obj_interest = obj
      break
    else:
      props = incomplete_scene_graph[obj]
      if len(props)<  5:  #len(properties):
        obj_interest = obj
        break

  incomplete_objects = copy.deepcopy(objects)
  incomplete_blender_objects = copy.deepcopy(blender_objects)


  incomplete_objects.pop(obj_interest)
  incomplete_blender_objects.pop(obj_interest)

  return incomplete_objects, incomplete_blender_objects
  

##---------------------------------------------------------------------------------------------------------------------------

def make_scene_empty(blender_objects):
  for obj in blender_objects:
    utils.delete_object(obj)

##---------------------------------------------------------------------------------------------------------------------------
def get_blender_objects(objects, objects_blender_info, blender_obj):
  blender_objects = []
  for index, obj_blender_info in enumerate(objects_blender_info):

    obj, pixel_coords  = blender_obj.add_object(args.shape_dir, 
      obj_blender_info['mat_name'], 
      obj_blender_info['obj_name'], 
      obj_blender_info['r'], 
      obj_blender_info['x'], 
      obj_blender_info['y'], 
      obj_blender_info['theta'],
      obj_blender_info['rgba'])  

    blender_objects.append(obj)
    objects[index]['pixel_coords'] = pixel_coords
    objects[index]['3d_coords'] = tuple(obj.location)

  return objects, blender_objects
          

##---------------------------------------------------------------------------------------------------------------------------


def get_constraint_types():
  with open(args.constraints_json, 'r') as f_constraints:
    constraint_types = json.load(f_constraints)

  #return constraint_types
  return list(constraint_types.values())


def balance_constraint_type(constraints_types_list, num_image_per_constraint_type, max_number_of_constraints_per_image):
  for i in constraints_types_list:
     if num_image_per_constraint_type[i] < max_number_of_constraints_per_image:
      return i
  return None  

  


# to make sure constraint types are equally distributed among images
def select_constraint_type(constraints_types_list, num_image_per_constraint_type, max_number_of_constraints_per_image):
  a = [i for i in range(len(num_image_per_constraint_type)) if num_image_per_constraint_type[i] < max_number_of_constraints_per_image]

  if len(a) > 0:
    constraint_type_index = random.choice(a)
  else:
    constraint_type_index = None

  return constraint_type_index


##---------------------------------------------------------------------------------------------------------------------------


def get_sorted_list(dict):
    sorted_dictionary = collections.OrderedDict(sorted(dict.items()))
    return list(sorted_dictionary.items())

##---------------------------------------------------------------------------------------------------------------------------


def add_objects(scene_struct, args, properties, complete_scene_graph):
  objects_blender_info = []

  positions = []
  objects = []

  num_objects = len(complete_scene_graph.keys())

  for i in range(num_objects):

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    
    while True:

      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        return add_objects(scene_struct, args, properties, complete_scene_graph)

      
      #x = random.uniform(-3.5, 3.5)
      #y = random.uniform(-3.5, 3.5)

      region_index = complete_scene_graph[i]['region']


      x1 = properties['regions'][region_index]['x'][0]
      x2 = properties['regions'][region_index]['x'][1]
      y1 = properties['regions'][region_index]['y'][0]
      y2 = properties['regions'][region_index]['y'][1]
      x = random.uniform(x1, x2)
      y = random.uniform(y1, y2)
      
      """
      print('region_index: ', region_index)
      print('object: ', i)
      print('ranges: ', x1, x2, y1, y2)
      print('x={} , y={}'.format(x, y))
      print(complete_scene_graph[i]['shape'], complete_scene_graph[i]['color'], complete_scene_graph[i]['size'], complete_scene_graph[i]['material'])
      print('-------------------------------------------')
      """
   

      # Check to make sure the new object is further than min_dist from all
      # other objects, and further than margin along the four cardinal directions
      dists_good = True
      margins_good = True
      for (xx, yy, rr) in positions:
        dx, dy = x - xx, y - yy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist - r - rr < args.min_dist:
          dists_good = False
          break

        for direction_name in ['left', 'right', 'front', 'behind']:
          direction_vec = scene_struct['directions'][direction_name]
          assert direction_vec[2] == 0
          margin = dx * direction_vec[0] + dy * direction_vec[1]
          if 0 < margin < args.margin:
            print(margin, args.margin, direction_name)
            print('BROKEN MARGIN!')
            #margins_good = False
            #break
        if not margins_good:
          break

      if dists_good and margins_good:
        break

      
    shape_name = complete_scene_graph[i]['shape']
    color_name = complete_scene_graph[i]['color']
    size_name = complete_scene_graph[i]['size']
    mat_name = complete_scene_graph[i]['material']

    
    r = properties['size'][size_name]
    rgba = [float(c) / 255.0 for c in properties['color'][color_name]] + [1.0]
    # For cube, adjust the size a bit
    if shape_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    #theta = 360.0

    objects_blender_info.append(
        {'obj_name': properties['shape'][shape_name], 
          'r': r, 'x': x, 'y':y, 'theta': theta, 
          'mat_name': properties['material'][mat_name], 
          'rgba': rgba})
    
    positions.append((x, y, r))
    objects.append({
      'shape': shape_name,
      'size': size_name,
      'material': mat_name,
      'rotation': theta,
      'color': color_name,
      'region': region_index
    })

    
  return objects, objects_blender_info

##---------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    main(args)
  elif '--help' in sys.argv or '-h' in sys.argv:
    parser.print_help()
  else:
    print('This script is intended to be called from blender like this:')
    print()
    print('blender --background --python render_images.py -- [args]')
    print()
    print('You can also run as a standalone python script to view all')
    print('arguments like this:')
    print()
    print('python render_images.py --help')  

    ## blender --background -noaudio --python render_images.py -- --num_images 1
    ## blender --background -noaudio --python render_images.py -- --num_images 3 --use_gpu 0 --start_idx 0