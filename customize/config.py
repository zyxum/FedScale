import yaml

global cfg
if 'cfg' not in globals():
    with open('/users/yuxuanzh/FedScale/customize/config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

