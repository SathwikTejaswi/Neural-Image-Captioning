import pickle
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image

#annotation_path   = 'G:\\uiuc\\sem3\\IE 534\\project\\results_20130124.token'
#root = 'G:\\uiuc\\sem3\\IE 534\\project\\filcker30k'
#
#annotations = pd.read_table(annotation_path , sep='\t', header=None, names=['image', 'caption'])
#annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
#annotations['image'] = annotations['image'].map(lambda x: x.split('#')[0])
#annotations = annotations[ annotations['image_num'] == '1' ]  
##index = 1
#caption = annotations['caption'][index]
#img_id = annotations['image'][index]
#image = Image.open(os.path.join(root, img_id)).convert('RGB')

# https://github.com/jazzsaxmafia/show_attend_and_tell/blob/master/make_flickr_dataset.py

class FlickerDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, token, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            token: coco annotation file path ( in this case results_20130124.token) .
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.annotation_path = token
        self.vocab = vocab
        self.transform = transform
        self.annotations = pd.read_table(self.annotation_path , sep='\t', header=None, names=['image', 'caption'])
        self.annotations['image_num'] = self.annotations['image'].map(lambda x: x.split('#')[1])
        self.annotations['image'] = self.annotations['image'].map(lambda x: x.split('#')[0])

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        annotations = self.annotations
        vocab = self.vocab
        caption = annotations['caption'][index]
        img_id = annotations['image'][index]
        image = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    Flicker = FlickerDataset(root=root,
                       token=token,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=Flicker, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader






