import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import *
from data_loader import get_loader 
from torchvision import transforms
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(arg):
    # Create model directory
    if not os.path.exists(arg['model_path']):
        os.makedirs(arg['model_path'])
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([transforms.RandomCrop(arg['crop_size']), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(arg['vocab_path'], 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(arg['image_dir'], arg['caption_path'], vocab, 
                             transform, arg['batch_size'],
                             shuffle=True, num_workers=arg['num_workers']) 

    # Build the models
    encoder = EncoderCNN(arg['embed_size']).to(device)
    decoder = DecoderRNN(arg['embed_size'], arg['hidden_size'], len(vocab), arg['num_layers']).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=arg['learning_rate'])
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(arg['num_epochs']):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions, lengths)
            loss = criterion(outputs, targets)
            decoder.zero_grad()
            encoder.zero_grad()
            loss.backward()
            optimizer.step()

            # Print log info
            if i % arg['log_step'] == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, arg['num_epochs'], i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % arg['save_step'] == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    arg['model_path'], 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    arg['model_path'], 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


if __name__ == '__main__':
    
    arg = {}
    print('Use defualt args?(Y/N)')
    ans = input()
    
    if ans == 'Y':
        arg = get_default()
    else :
        arg = get_args()

    main(arg)
