from fedscale.core.arg_parser import args
from customized_models import resnet34, shufflenet_v2_x2_0

outputClass = {'Mnist': 10, 'cifar10': 10, "imagenet": 1000, 'emnist': 47,'amazon':5,
                'openImg': 596, 'google_speech': 35, 'femnist': 62, 'yelp': 5, 'inaturalist' : 1010
            }

def customized_init_model():
    if args.model == 'resnet34':
        return resnet34(num_classes=outputClass[args.data_set])
    elif args.model == 'shufflenet_v2_x2_0':
        return shufflenet_v2_x2_0(num_classes=outputClass[args.data_set])