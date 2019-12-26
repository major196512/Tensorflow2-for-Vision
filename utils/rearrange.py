import tensorflow as tf

def resnet_weight(pretrain_dir):
    def conv_weight(d, k, t):
        if k[0] == 'weights' : key = k[0]
        elif k[0] == 'BatchNorm': key = k[1]
        d[key] = t

    new_dict = dict()

    checkpoint = tf.train.load_checkpoint(pretrain_dir)
    for k, v in checkpoint.get_variable_to_dtype_map().items():
        checkpoint_tensor = checkpoint.get_tensor(k)
        key = k.split('/')
        if key[0] == 'global_step' : continue

        block_name = key[1]
        if block_name not in new_dict : new_dict[block_name] = dict()

        if block_name == 'conv1':
            conv_weight(new_dict[block_name], key[2:], checkpoint_tensor)

        elif 'block' in block_name:
            unit_num = int(key[2].split('_')[1])
            conv_name = key[4]
            if unit_num not in new_dict[block_name] :
                new_dict[block_name][unit_num] = dict()
            if conv_name not in new_dict[block_name][unit_num] :
                new_dict[block_name][unit_num][conv_name] = dict()
            conv_weight(new_dict[block_name][unit_num][conv_name], key[5:], checkpoint_tensor)

        elif block_name == 'logits':
            new_dict[block_name][key[2]] = checkpoint_tensor

    new_dict['logits']['weights'] = new_dict['logits']['weights'].reshape(-1, 1000)

    return new_dict
