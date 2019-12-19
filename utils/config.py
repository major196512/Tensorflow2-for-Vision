import yaml

def merge_yaml(input):
    cfg = yaml.load(open('./configs/base.yaml'), Loader=yaml.FullLoader)
    #################################
    ####DATASET
    #################################
    cfg_dataset = cfg['DATA']
    dataset = input['DATA']['dataset']

    if dataset in cfg_dataset['data_dir']:
        cfg_dataset['data_dir'] = cfg_dataset['data_dir'][dataset]
    else:
        raise ValueError('Not in base yaml for data directory')

    #################################
    ####Merging
    #################################
    for base_key in input.keys():
        for k1 in input[base_key].keys():
            if isinstance(input[base_key][k1], dict):
                for k2 in input[base_key][k1].keys():
                    cfg[base_key][k1][k2] = input[base_key][k1][k2]

            else:
                cfg[base_key][k1] = input[base_key][k1]

    return cfg
