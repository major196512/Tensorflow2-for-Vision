def merge_yaml(cfg, input):
    #################################
    ####DATASET
    #################################
    cfg_dataset = cfg['DATASET']
    dataset = input['DATASET']
    data = dataset['data']

    if dataset['data'] in cfg_dataset['data_dir']:
        cfg_dataset['data_dir'] = cfg_dataset['data_dir'][data]
    else:
        raise ValueError('Not in base yaml for data directory')

    #################################
    ####Merging
    #################################
    for base_key in input.keys():
        for k1 in input[base_key].keys():
            if isinstance(input[base_key][k1], dict):
                for k2 in input[base_key][k1].keys():
                    dfg[base_key][k1][k2] = input[base_key][k1][k2]

            else:
                cfg[base_key][k1] = input[base_key][k1]

    return cfg
