import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import configparser
from utils.data_loader import get_loader 
from utils.build_vocab import Vocabulary
from model_script.model3 import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    params = config['TRAIN3']
    model_path = params['model_path']
    crop_size = int(params['crop_size'])
    vocab_path = params['vocab_path']
    image_dir = params['image_dir']
    caption_path = params['caption_path']
    log_step = int(params['log_step'])
    save_step = int(params['save_step'])
    embed_size = int(params['embed_size'])
    hidden_size = int(params['hidden_size'])
    num_layers = int(params['num_layers'])
    num_epochs = int(params['num_epochs'])
    batch_size = int(params['batch_size'])
    num_workers = int(params['num_workers'])
    learning_rate = float(params['learning_rate'])
    


    # Create model directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(image_dir, caption_path, vocab, 
                             transform, batch_size,
                             shuffle=True, num_workers=num_workers) 

    # Build the models
    encoder = EncoderCNN(embed_size).cuda()
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).cuda()
    
    # load weights
    #encoder.load_state_dict(torch.load('encoder-5-3000.ckpt'))
    #decoder.load_state_dict(torch.load('decoder-5-3000.ckpt'))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    
    # Train the models
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, captions, lengths) in enumerate(data_loader):
            
            # Set mini-batch dataset
            images = images.cuda()
            captions = captions.cuda()
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
            if i % log_step == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                      .format(epoch, num_epochs, i, total_step, loss.item(), np.exp(loss.item()))) 
                
            # Save the model checkpoints
            if (i+1) % save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    model_path, 'v3_decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    model_path, 'v3_encoder-{}-{}.ckpt'.format(epoch+1, i+1)))


main()
