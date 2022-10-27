# preprocess clevr questions
# code adopted from https://github.com/facebookresearch/clevr-iep/blob/master/scripts/preprocess_questions.py

"""
one_hop = 4
same_relate = 11
single_and =  4
three_hop = 3 
two_hop = 3 
zero_hop = 3
"""

import sys
sys.path.append("/content/drive/MyDrive/CLEVR-ABDUCTIVE/nesy/ns-vqa-master/reason")
import os
import argparse
import json

import h5py
import numpy as np

import utils.programs as program_utils
import utils.preprocess as preprocess_utils
import utils.utils as utils

print('a')
parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='chain',
                    choices=['chain', 'prefix', 'postfix'])
parser.add_argument('--input_questions_json', required=True)
parser.add_argument('--input_vocab_json', default='')
parser.add_argument('--expand_vocab', default=0, type=int)
parser.add_argument('--unk_threshold', default=1, type=int)
parser.add_argument('--encode_unk', default=0, type=int)

parser.add_argument('--output_h5_file', required=True)
parser.add_argument('--output_vocab_json', default='')
parser.add_argument('--input_scenes_json', required=True)


def program_to_str(program, mode):
    if mode == 'chain':
        if not program_utils.is_chain(program):
            return None
        return program_utils.list_to_str(program)
    elif mode == 'prefix':
        program_prefix = program_utils.list_to_prefix(program)
        return program_utils.list_to_str(program_prefix)
    elif mode == 'postfix':
        program_postfix = program_utils.list_to_postfix(program)
        return program_utils.list_to_str(program_postfix)
    return None




def read_files(file_path):
   with open(file_path, 'r') as file:
      print(file.read())

def read_labels():
        with open('../data/properties.json', encoding="utf-8") as f:
            properties = json.load(f)
        sorted_key_properties = sorted(properties.keys())

        key_properties_values = []
        for key_property in sorted_key_properties:
            key_properties_values.extend(sorted(properties[key_property].keys()))
        labels = {k: v for v, k in enumerate(key_properties_values)}
        return labels

def main(args):

    labels = read_labels()
    
    # Opening JSON file
    #f = open('/home/savitha/Documents/ns-vqa-master/data/raw/CLEVR_v1.0/questions/CLEVR_test_questions.json', 'r')
    #questions = json.load(f)['questions']
    #print(questions[0])
    # returns JSON object as
    # a dictionary
    #data = json.load(f)
 
    # Iterating through the json
    # list
    #for i in data['emp_details']:
     #   print(i)
 
    # Closing file
    #f.close()
    
    if (args.input_vocab_json == '') and (args.output_vocab_json == ''):
        print('Must give one of --input_vocab_json or --output_vocab_json')
        return

    print('Loading data')
    
    path = args.input_questions_json
    os.chdir(path)
    
    questions = []
    for file in os.listdir():
      # Create the filepath of particular file
        file_path =f"{path}/{file}"
        with open(file_path, 'r') as f:
          question_dict  = json.load(f)
          questions.append(question_dict)
    
    path = args.input_scenes_json
    os.chdir(path)
    scenes = []
    for file in os.listdir():
      # Create the filepath of particular file
        file_path =f"{path}/{file}"
        with open(file_path, 'r') as f:
          scene_dict  = json.load(f)
          scenes.append(scene_dict)
    

    #with open(args.input_questions_json, 'r') as f:
    #    questions = json.load(f)['questions']
    #print("Loading:", questions)
    # Either create the vocab or load it from disk
    if args.input_vocab_json == '' or args.expand_vocab == 1:
        print('Building vocab')
        #if 'answer' in questions[0]:
        #    answer_token_to_idx = preprocess_utils.build_vocab(
        #        (q['answer'] for q in questions)
        #    )
        question_token_to_idx = preprocess_utils.build_vocab(
            (q['question'] for q in questions),
            min_token_count=args.unk_threshold,
            punct_to_keep=[';', ','], punct_to_remove=['?', '.']
        )
        all_program_strs = []
        for q in questions:
            if 'asp_query' not in q: continue
            program_str = q['asp_query']
            #program_str = program_to_str(q['asp_query'], args.mode)
            #print(program_str)
            if program_str is not None:
                all_program_strs.append(program_str)
        program_token_to_idx = preprocess_utils.build_vocab_program(all_program_strs, punct_to_keep=[';', ',', '.', '(', ')',':-', '!=' ])
        
        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'program_token_to_idx': program_token_to_idx,
            'labels': labels
        }
        print("VOCAB::", vocab)
    if args.input_vocab_json != '':
        print('Loading vocab given..')
        if args.expand_vocab == 1:
            new_vocab = vocab
        with open(args.input_vocab_json, 'r') as f:
            vocab = json.load(f)
        if args.expand_vocab == 1:
            num_new_words = 0
            for word in new_vocab['question_token_to_idx']:
                if word not in vocab['question_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['question_token_to_idx'])
                    vocab['question_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Found %d new words' % num_new_words)
            for word in new_vocab['program_token_to_idx']:
                if word not in vocab['program_token_to_idx']:
                    print('Found new word %s' % word)
                    idx = len(vocab['program_token_to_idx'])
                    vocab['program_token_to_idx'][word] = idx
                    num_new_words += 1
            print('Found %d new words' % num_new_words)
            #for word in new_vocab['answer_token_to_idx']:
            #    if word not in vocab['answer_token_to_idx']:
            #        print('Found new word %s' % word)
            #        idx = len(vocab['answer_token_to_idx'])
            #        vocab['answer_token_to_idx'][word] = idx
            #        num_new_words += 1
            #print('Found %d new words' % num_new_words)

    if args.output_vocab_json != '':
        utils.mkdirs(os.path.dirname(args.output_vocab_json))
        with open(args.output_vocab_json, 'w') as f:
            json.dump(vocab, f)

    # Encode all questions and programs
    print('Encoding data')
    questions_encoded = []
    programs_encoded = []
    question_families = []
    #complete_image_index = []
    constraint_type = []
    orig_idxs = []
    image_idxs = []
    answers = []
    temp_file = {}
    num_fly = 1
    for orig_idx, q in enumerate(questions):
        #print(q)
        question = q['question']
        t_f = q['template_filename']
        orig_idxs.append(orig_idx)
        image_idxs.append(q['image_index'])
        constraint_type.append(int(scenes[orig_idx]['constraint_type_index']))
        if 'question_family_index' in q:
            qfi = q['question_family_index']
            if (t_f, qfi) not in temp_file:
              temp_file[(t_f, qfi)] = num_fly
              num_fly = num_fly+1 
            
            question_families.append(temp_file[(t_f, qfi)])
            #question_families.append(q['question_family_index'])
        question_tokens = preprocess_utils.tokenize(question,
                                                punct_to_keep=[';', ','],
                                                punct_to_remove=['?', '.'])
        
        
        question_encoded = preprocess_utils.encode(question_tokens,
                                                 vocab['question_token_to_idx'],
                                                 allow_unk=args.encode_unk == 1)
        questions_encoded.append(question_encoded)

        #if 'program' in q:
        #    program = q['program']
        #    program_str = program_to_str(program, args.mode)
        #    program_tokens = preprocess_utils.tokenize(program_str)
        #    program_encoded = preprocess_utils.encode(program_tokens, vocab['program_token_to_idx'])
        #    programs_encoded.append(program_encoded)

        if 'asp_query' in q:
            program_str = q['asp_query']
            program_tokens = preprocess_utils.tokenize_program(program_str, punct_to_keep=[';', ',', '.', '(', ')',':-', '!=' ])
            program_encoded = preprocess_utils.encode(program_tokens, vocab['program_token_to_idx'])
            programs_encoded.append(program_encoded)

        if 'answer' in q:
            a = [labels[i] for i in q['answer']]
            b = [1 if i in a else 0 for i in range(len(labels))]  
            answers.append(b)
    
    # Pad encoded questions and programs
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    if len(programs_encoded) > 0:
        max_program_length = max(len(x) for x in programs_encoded)
        for pe in programs_encoded:
            while len(pe) < max_program_length:
                pe.append(vocab['program_token_to_idx']['<NULL>'])


    # Create h5 file
    print('##############Writing output.....')
    print(temp_file)
    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    programs_encoded = np.asarray(programs_encoded, dtype=np.int32)
    print(questions_encoded.shape)
    print(programs_encoded.shape)
    utils.mkdirs(os.path.dirname(args.output_h5_file))
    with h5py.File(args.output_h5_file, 'w') as f:
        f.create_dataset('questions', data=questions_encoded)
        f.create_dataset('image_idxs', data=np.asarray(image_idxs))
        f.create_dataset('orig_idxs', data=np.asarray(orig_idxs))
        
        f.create_dataset('constraint_type', data=np.asarray(constraint_type))
        if len(programs_encoded) > 0:
            f.create_dataset('programs', data=programs_encoded)
        if len(question_families) > 0:
            f.create_dataset('question_families', data=np.asarray(question_families))
        if len(answers) > 0:
            f.create_dataset('answers', data=np.asarray(answers))
        #f.create_dataset('vocab', data=vocab)
        #print(f)
        f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
