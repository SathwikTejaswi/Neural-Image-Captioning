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
    arg['model_path'] = 
    arg['crop_size'] = 
    arg['vocab_path'] = 
    arg['image_dir'] = 
    arg['caption_path'] = 
    arg['log_step'] = 
    arg['save_step'] = 
    arg['embed_size'] = 
    arg['hidden_size'] = 
    arg['num_layers'] = 
    arg['num_epochs'] = 
    arg['batch_size'] = 
    arg['num_workers'] = 
    arg['learning_rate'] = 
    
    return(arg)
