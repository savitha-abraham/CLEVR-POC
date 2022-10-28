import os, sys
import json
from typing import Counter
from pathlib import Path
import numpy

sys.path.append(os.path.join(Path(__file__).parents[1], 'options'))
sys.path.append(os.path.join(Path(__file__).parents[1]))

from executors.aspsolver import solve, getToken
from test_options import TestOptions
from datasets import get_dataloader
from models.parser import Seq2seqParser
import utils.utils as utils
import torch


import warnings


def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True

if torch.cuda.is_available():
  print("cuda available..")  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev)
warnings.filterwarnings("ignore")

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
model = Seq2seqParser(opt).to(device)

if (opt.load_checkpoint_path is not None):
  checkpoint = torch.load(opt.load_checkpoint_path)
  
  
print('| running test')
stats = {
    'correct_ans': 0,
    'correct_prog': 0,
    'total': 0
}


vocab = utils.load_vocab(opt.clevr_vocab_path)
test_scene_path = opt.clevr_val_scene_path
env_folder =  opt.clevr_constraint_scene_path

for x, y, answer, idx,constraint_type in loader:
    x = x.to(device = device)
    y = y.to(device = device)
    model.set_input(x, y)
    programs = model.parse()
    pg_np = programs.cpu().detach().numpy()
    y_np = y.cpu().detach().numpy()
    ans_np = answer.to(device = device)
    ct_np = constraint_type.cpu().detach().numpy()
    
    for i in range(pg_np.shape[0]): 
        ans = ans_np[i]
        ans_tokens = [vocab["labels_idx_to_token"][j]  for j, ab in enumerate(list(ans)) if ans[j]==1]
        ans_tokens_str = ' '.join(ans_tokens)
        pred_pgm = getToken(pg_np[i], vocab['program_idx_to_token'])
        pred= solve(pred_pgm, idx[i],  ct_np[i], 'val', test_scene_path, env_folder)
        if pred != None:
            a = [vocab['labels'][d] for d in pred]
            b = [1 if c in a else 0 for c in range(len(vocab['labels']))]
            predicted = numpy.array(b)
            if numpy.array_equal(predicted, ans):
                stats['correct_ans']+=1
            
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1
       
        stats['total'] += 1
        
        
    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

result = {
    'program_acc': stats['correct_prog'] / stats['total'],
    'overall_acc': stats['correct_ans'] / stats['total'],
    'correct-ans':stats['correct_ans'],
}
print(result)
utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    
