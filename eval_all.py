import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
import nltk
import configparser
from utils.pycocoeval import *
from torchvision import transforms
from utils.data_loader import get_loader 
from utils.build_vocab import Vocabulary
from model_script.model import EncoderCNN, DecoderRNN
from PIL import Image

c = nltk.translate.bleu_score.SmoothingFunction()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def main():


    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['EVAL']
    encoder_path = params['encoder_path']
    decoder_path = params['decoder_path']
    crop_size = int(params['crop_size'])
    vocab_path = params['vocab_path']
    image_dir = params['image_dir']
    caption_path = params['caption_path']
    embed_size = int(params['embed_size'])
    hidden_size = int(params['hidden_size'])
    num_layers = int(params['num_layers'])
    batch_size = int(params['batch_size'])
    num_workers = int(params['num_workers'])

    # Image preprocessing
    transform = transforms.Compose([ 
        transforms.Resize(229),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size,hidden_size, len(vocab), num_layers).eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    data_loader = get_loader(image_dir,caption_path,vocab,transform,batch_size,True,num_workers)
    
    bleu_score = 0
    

    def id_to_word(si):
        s = []
        for word_id in si:
            word = vocab.idx2word[word_id]
            s.append(word)
            if word == '<end>':
                break
        return(s)




    for i, (images, captions, lengths) in enumerate(data_loader):
        # Generate an caption from the image
        images = images.to(device)
        feature = encoder(images)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        captions = captions.detach().cpu().numpy()
        references = []
        for cap in captions:
            references.append(id_to_word(cap))
        gen_cap = id_to_word(sampled_ids)
        
        rng = range(1)
        
        res = {0: [{'image_id': 0, 'caption': ' '.join(gen_cap)}]}
        gts = {0: [{'image_id': 0, 'caption': ' '.join(references[0])}]}
        evalObj = COCOEvalCap(rng,gts,gts)
        evalObj.evaluate()        
        print(evalObj.eval)
        
main()        
