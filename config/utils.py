import yaml

def parse_cfg(cfg_path):
    # cfg = {}
    # with open(cfg_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line[0] == '#' or line == '\n':
    #             continue
    #         line = line.strip().split(':')
    #         key, value = line[0].strip(), line[1].strip()
    #         cfg[key] = value

    with open(cfg_path, 'r') as f:
        # firstline=f.readline()
        # print(firstline)
        # if firstline[0] == '!':
        #     next(f) # 第一行不知道啥玩意
        cfg = yaml.unsafe_load(f)
        # cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict
    return cfg

def parse_dump(cfg_path,ydict,mode='w+'):
    with open(cfg_path,mode) as f:
        yaml.dump(ydict,f)