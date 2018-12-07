import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image
import pandas as pd
from pycocotools.coco import COCO
import json
from nltk.translate.bleu_score import corpus_bleu

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    actual_captions = []
    predicted_captions = []
    annotation_path = '../data/annotations/captions_val2014.json'
    with open(annotation_path) as f:
        anns = json.load(f)
    anns = anns["annotations"]
    for index,_ in enumerate(anns):
        anns[index]['image_id'] = str(anns[index]['image_id']).rjust(12, '0')
        if anns[index]['caption'][-1] == '.':
            #print (anns[index]['caption'])
            anns[index]['caption'] = str(anns[index]['caption'])[:-1]

    anns = pd.DataFrame(anns)
    #print (anns.head())

    for index, image_name in enumerate(os.listdir("../data/val2014/")):
        try:
            print (index)
            image = load_image("../data/val2014/"+image_name, transform)
            image_tensor = image.to(device)
        
            feature = encoder(image_tensor)
            sampled_ids = decoder.sample(feature)
            sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
            
            sampled_caption = []
            for word_id in sampled_ids:
                word = vocab.idx2word[word_id]
                sampled_caption.append(word)
                if word == '<end>':
                    break
            sentence = ' '.join(sampled_caption)
            #print (sampled_caption)
            sampled_caption = sampled_caption[1:-2]
            #print (sampled_caption)
            predicted_captions.append(sampled_caption)
            #print (image_name)
            image_id = image_name[-16:-4]
            #print (image_id)
            temp = anns[anns['image_id'] == image_id]
            actual = [i.split(' ') for i in temp['caption']]
            #print (actual)
            actual_captions.append(actual)
        except RuntimeError:
            print (image_name + " errored out")
            pass

    pickle.dump(predicted_captions, open("predicted_captions.p", 'wb'))
    pickle.dump(actual_captions, open("actual_captions.p", 'wb'))
    one_reference = [cap[0] for cap in actual_captions]
    pickle.dump(one_reference, open("one_reference_actual.p",'wb'))
    print (corpus_bleu(one_reference,predicted_captions))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
