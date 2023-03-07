# Code from https://github.com/t0efL/end2end-HKR-research
from omegaconf import OmegaConf

# cyrillic synthetic ' !"%\'()+,-./0123456789:;=?R[]bcehioprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№'
# hkr synthetic ' !"%(),-.0123456789:;?HoАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…'
# cyrillic hkr synthetic ' !"%\'()+,-./0123456789:;=?HR[]bcehioprstuxy«»АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…№'

config = {
    'seed': 0xFACED,
    'training': {
        'num_epochs': 100,
        'early_stopping': 10,
        'device': 'cpu'
    },
    'paths': {
        'datasets': {
            'original': 'img',
            'stackmix': None,  # path to parts
            'letters': None,
            'generate': '/home/dedoc/hrtr/attention/fonts',  # path to fonts to generate data
        },
        'path_to_train_labels': 'annotations/train_hkr.json',
        'path_to_val_labels': 'annotations/val_hkr.json',
        'save_dir': 'out',
        'path_to_checkpoint': None,
        'path_to_pretrain': None
    },
    'data': {
        'alphabet': ' "\'%0123456789!(),+-.:;?/=R[]№bcehioprstuxy«»HoАБВГДЕЖЗИЙКЛМНОПРСТУФХЧШЩЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяёғҚқҮӨө–—…',
        'dataloader_params': {
            'batch_size': 64,
            'num_workers': 8,
            'pin_memory': False,
            'persistent_workers': True,
        },
        'training_sizes': {
            'samples_per_epoch': 60000,
            'proportions': {
                'original_train': 1,
                'stackmix': 0,
                'letters': 0,
                'generate': 0
            }
        }
    },
    'model_params': {
        'transformer_decoding_params': {
            'max_new_tokens': 30,
            'min_length': 1,
            'num_beams': 1,
            'num_beam_groups': 1,
            'do_sample': False
        }
    },
    'logging': {
        'log': False,
        'wandb_username': 'toefl',
        'wandb_project_name': 'nto'
    },
    'ctc_decode': {
        'beam_search': False,
        'lm_path': None
    }
}
config = OmegaConf.create(config)
