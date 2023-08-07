import json
import os
import pickle
import argparse
import numpy as np
import string
from tqdm import tqdm

def sentence_generation(region_gt, output_path):
    """
    Visual Genome Region Descriptions data file generatior
    """
    output_file = os.path.join(output_path, "sentences.pkl")
    
    sentences = []
    for i, regs in enumerate(tqdm(region_gt)):
        reg_gt = regs['regions']
        for j, reg in enumerate(reg_gt):
            sen = words_preprocess(reg['phrase'])
            sentences.append(sen)

    dist_senteces = list(set(sentences))
    """for value in tqdm(sentences):
        if value not in dist_senteces:
            dist_senteces.append(value)"""
            
    with open(output_file,"wb") as pkl_file:
        pickle.dump(dist_senteces, pkl_file)
    print(f'number of sentences : {len(dist_senteces)}')
    
    return dist_senteces


def words_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    translator = str.maketrans('', '', string.punctuation)
    replacements = {
        u'½': u'half',
        u'—': u'-',
        u'™': u'',
        u'¢': u'cent',
        u'ç': u'c',
        u'û': u'u',
        u'é': u'e',
        u'°': u' degree',
        u'è': u'e',
        u'…': u'',
    }
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(translator)


def main(args):
    print("data loading...")
    with open(args.region_data, 'r') as f:
        region_data = json.load(f)
    """with open(args.split_json, 'r') as f:
        split_data = json.load(f)
    with open(args.image_data, 'r') as f:
        image_data = json.load(f)"""
        
    sentences = sentence_generation(region_data, args.pickle_output)
    
    #all_image_ids = [image_data[i]['image_id'] for i in range(len(image_data))]
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # INPUT settings
    parser.add_argument('--region_data',
                        default='data/visual-genome/region_descriptions.json',
                        help='Input JSON file with regions and captions')
    parser.add_argument('--image_data',
                        default='data/visual-genome/image_data.json',
                        help='Input JSON file with image url weight and height')
    parser.add_argument('--split_json',
                        default='info/densecap_splits.json',
                        help='JSON file of splits')

    # OUTPUT settings
    parser.add_argument('--pickle_output',
                        default='data',
                        help='Path to output pickle file')
    parser.add_argument('--h5_output',
                        default='data/VG-regions-lite.h5',
                        help='Path to output HDF5 file')

    # OPTIONS
    parser.add_argument('--image_size',
                        default=720, type=int,
                        help='Size of longest edge of preprocessed images')
    parser.add_argument('--max_token_length',
                        default=15, type=int,
                        help="Set to 0 to disable filtering")
    parser.add_argument('--min_token_instances',
                        default=15, type=int,
                        help="When token appears less than this times it will be mapped to <UNK>")
    parser.add_argument('--tokens_type', default='words',
                        help="Words|chars for word or char split in captions")
    parser.add_argument('--max_images', default=-1, type=int,
                        help="Set to a positive number to limit the number of images we process")
    args = parser.parse_args()
    main(args)