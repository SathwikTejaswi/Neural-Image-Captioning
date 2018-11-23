def get_args():
    print('please provide the following :')
    arg = {}
    
    print('model path :')
    arg['model_path'] = input()
    
    print('crop size :')
    arg['crop_size'] = int(input())
    
    print('vocab path :')
    arg['vocab_path'] = input()
    
    print('image dir :')
    arg['image_dir'] = input()
    
    print('caption path :')
    arg['caption_path'] = input()
    
    print('log_step :')
    arg['log_step'] = input()
    
    print('save_step :')
    arg['save_step'] = int(input())
    
    print('embed_size :')
    arg['embed_size'] = int(input())
    
    print('hidden_size :')
    arg['hidden_size'] = int(input())
    
    print('num_layers :')
    arg['num_layers'] = int(input())
    
    print('num_epochs :')
    arg['num_epochs'] = int(input())
    
    print('batch_size :')
    arg['batch_size'] = int(input())
    
    print('num_workers :')
    arg['num_workers'] = int(input())
    
    print('learning_rate :')
    arg['learning_rate'] = int(input())
    
    return(args)

def get_default():
    arg = {}
    arg['model_path'] = 'models/'
    arg['crop_size'] = 224
    arg['vocab_path'] = 'data/vocab.pkl'
    arg['image_dir'] = 'data/resized2014'
    arg['caption_path'] = 'data/annotations/captions_train2014.json'
    arg['log_step'] = 10
    arg['save_step'] = 1000
    arg['embed_size'] = 256
    arg['hidden_size'] = 512
    arg['num_layers'] = 1
    arg['num_epochs'] = 5
    arg['batch_size'] = 128
    arg['num_workers'] = 2
    arg['learning_rate'] = 0.001
    
    return(arg)
