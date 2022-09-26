import math, sys, random, argparse, json, os, tempfile
import collections 

from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from image_generation import scene_info, blender
from generate_dataset import parser
from generate_environment import generateEnvironment


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


##---------------------------------------------------------------------------------------------------------------------------

def main(args):
  
  image_dir = os.path.join(args.complete_data_dir, args.image_dir, args.split)
  scene_dir = os.path.join(args.complete_data_dir, args.scene_dir)
  environment_constraints_dir = os.path.join(args.incomplete_data_dir, args.environment_constraints_dir)

  if not os.path.isdir(image_dir):
    os.makedirs(image_dir) 
  if not os.path.isdir(scene_dir):
    os.makedirs(scene_dir) 
  if not os.path.isdir(environment_constraints_dir):
    os.makedirs(environment_constraints_dir)     




  num_digits = 6

  prefix = '%s_' % (args.filename_prefix)
  image_temp = '%s%%0%dd.png' % (prefix, num_digits)
  img_template = os.path.join(image_dir, image_temp)
  

  
  all_scenes = get_already_rendered_scenes(split=args.split, scene_dir=scene_dir)

  constraints_types_list = []
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
  
  i = args.start_idx

  # we have in total 9 regions, and we do not want to have more than one object at each region
  #if args.max_objects > 9:
  #  args.max_objects = 9

  num_image_per_constraint_type = [0 for i in range(args.num_constraint_types)]
  max_number_of_images_per_constraint = math.floor(args.num_images/args.num_constraint_types)

  env_id = 0
  objNum_env = {i:[] for i in range(args.min_objects,args.max_objects+1)}
  
  possible_num_objects = [i for i in range(args.min_objects, args.max_objects)]
  while i < args.num_images:
  #for i in range(args.num_images):
    img_path = img_template % i

    num_objects = random.choice(possible_num_objects)
    
    if (env_id < args.num_constraint_types):
      generateEnvironment(args, num_objects, env_id)
      objNum_env[num_objects].append(env_id) 
      num_image_per_constraint_type[env_id]= num_image_per_constraint_type[env_id] +1
      constraint_type_index = env_id
      env_id = env_id +1
    else:
      list_env = objNum_env[num_objects]
      constraint_type_index = balance_constraint_type(list_env, num_image_per_constraint_type, max_number_of_images_per_constraint)
      if constraint_type_index == None:
        possible_num_objects.remove(num_objects)
        continue
      


    

    

    #get a random constraint type from the list of available constraint types in the constraint.json file
    
    
    #constraint_type_index = random.choice(range(len(constraints_types_list)))
    #constraint_type = constraints_types_list[constraint_type_index]

    #constraint_type_index = select_constraint_type(constraints_types_list, num_image_per_constraint_type, max_number_of_images_per_constraint)
    #if constraint_type_index is not None:
    #    num_image_per_constraint_type[constraint_type_index] = num_image_per_constraint_type[constraint_type_index] + 1
    #    constraint_type = constraints_types_list[constraint_type_index]

    scene = render_scene(args,
      num_objects=num_objects,
      image_index=i,
      image_path=img_path,
      constraint_type_index=constraint_type_index,
      properties=properties
    )

    all_scenes.append(scene)

    i += 1
    if i == args.start_idx + args.render_batch_size:  #to avoid GPU CUDA overflow!
      break
    #else:
    #  break

  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  if len(all_scenes) > 0:
    output = {
        'info': {
          'date': args.date,
          'version': args.version,
          'split': args.split,
          'license': args.license,
        },
        'scenes': all_scenes
    }

    with open(os.path.join(scene_dir, args.split + '.json'), 'w') as f:
      json.dump(output, f)  

    print(num_image_per_constraint_type)

  else:
    print('EMPTY SCENES!!!')


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
    num_objects=5,
    image_index=0,
    image_path='render.png',
    constraint_type_index=None,
    properties=None
  ):

  blender_obj = blender.Blender(image_path, 
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
  scene_struct = {
      'image_index': image_index,
      'image_filename': os.path.basename(image_path),
      'objects': [],
      'directions': {},
  }


  plane_behind, plane_left, plane_up = blender_obj.get_plane_direction()

  # Save all six axis-aligned directions in the scene struct
  scene_struct['directions']['behind'] = tuple(plane_behind)
  scene_struct['directions']['front'] = tuple(-plane_behind)
  scene_struct['directions']['left'] = tuple(plane_left)
  scene_struct['directions']['right'] = tuple(-plane_left)
  scene_struct['directions']['above'] = tuple(plane_up)
  scene_struct['directions']['below'] = tuple(-plane_up)

  
  # Get region constraints from the constraints.json file
  regions = get_regions_info(constraint_type, properties)

  # Building a (complete) scene and check the validity and visibility of all the randomly added objects
  while (True):
    objects, objects_blender_info = add_objects(scene_struct, num_objects, args, regions, properties)
    objects, blender_objects = get_blender_objects(objects, objects_blender_info, blender_obj)
    all_visible = blender_obj.check_visibility(blender_objects, args.min_pixels_per_object)
 
    if not all_visible:
      # If any of the objects are fully occluded then start over; delete all
      # objects from the scene and place them all again.
      print('Some objects are occluded; replacing objects')
      make_scene_empty(blender_objects)        
    else:
      break


  scene_struct['objects'] = objects
  scene_struct['relationships'] = scene_info.compute_all_relationships(scene_struct)
  scene_struct['objects_blender_info'] = objects_blender_info
  scene_struct['constraint_type'] = constraint_type_index
  
  blender_obj.render()

  return scene_struct


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


def get_regions_info(constraint_type, properties):
  
  regions_info = constraint_type['regions']
  regions = []
  
  for idx, reg in enumerate(regions_info):   
      r = scene_info.Region(x_range=reg['range']['x'], 
                            y_range=reg['range']['y'],
                            index = idx,
                            constraints=reg['constraints'], 
                            properties=properties)
      regions.append(r)
  return regions

##---------------------------------------------------------------------------------------------------------------------------


def get_sorted_list(dict):
    sorted_dictionary = collections.OrderedDict(sorted(dict.items()))
    return list(sorted_dictionary.items())

##---------------------------------------------------------------------------------------------------------------------------


def add_objects(scene_struct, num_objects, args, regions, properties):
  objects_blender_info = []

  positions = []
  objects = []

  num_objects_per_region = [0 for i in range(len(regions))]
  

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
        return add_objects(scene_struct, num_objects, args, regions, properties)

      """
      x = random.uniform(-3.5, 3.5)
      y = random.uniform(-3.5, 3.5)
      region = scene_info.find_region(regions, x, y)
      """

      x = random.uniform(-3.5, 3.5)
      y = random.uniform(-3.5, 3.5)
      region = scene_info.find_region(regions, x, y)
      if region is not None:
        if num_objects_per_region[region.get_index()] != 0:
          region = None
      

      if region is not None:
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
              margins_good = False
              break
          if not margins_good:
            break

        if dists_good and margins_good:
          break
      else:
        print("Invalid region!")      
        # Choose color and shape according to the relevant region

    if region is not None:
      
      shape_name, color_name, size_name, mat_name = region.get_region_features()

      
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

      })
      num_objects_per_region[region.get_index()] = 1
      
    
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