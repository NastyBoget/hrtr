# Code from https://github.com/t0efL/end2end-HKR-research
from omegaconf import OmegaConf

config = {
    'seed': 0xFACED,
    'training': {
        'num_epochs': 100,
        'early_stopping': 10,
        'device': 'cuda'
    },
    'paths': {
        'datasets': {
            'original': '/home/dedoc/hrtr/transformer/img_large/img',
            'hkr': None,
            'stackmix': None,  # path to parts
            'kohtd': None,
            'letters': None,
            'kaggle': None,
            'synthetics': None,
        },
        'path_to_train_labels': '/home/dedoc/hrtr/transformer/annotations/train_cyrillic_large.json',
        'path_to_val_labels': '/home/dedoc/hrtr/transformer/annotations/test_cyrillic.json',
        'save_dir': '/home/dedoc/hrtr/transformer/cyrillic_large',
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
            'samples_per_epoch': 500000,
            'proportions': {
                'original_train': 1,
                'hkr': 0,
                'stackmix': 0,
                'kohtd': 0,
                'letters': 0,
                'kaggle': 0,
                'synthetics': 0
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
