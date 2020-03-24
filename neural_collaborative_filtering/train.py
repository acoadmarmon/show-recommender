import pandas as pd
import numpy as np
from data_loader.data_loader import SampleGenerator
from trainer.trainer import Trainer
from model.gmf import GMF
import argparse

gmf_config = {'alias': 'gmf_factor8neg4-implict',
              # 'optimizer': 'sgd',
              # 'sgd_lr': 1e-3,
              # 'sgd_momentum': 0.9,
              # 'optimizer': 'rmsprop',
              # 'rmsprop_lr': 1e-3,
              # 'rmsprop_alpha': 0.99,
              # 'rmsprop_momentum': 0,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_negative': 4,
              'l2_regularization': 0, # 0.01
              'use_cuda': False,
              'device_id': 0}

mlp_config = {'alias': 'mlp_factor8neg4_bz256_166432168_pretrain_reg_0.0000001',
              'num_epoch': 200,
              'batch_size': 256,  # 1024,
              'optimizer': 'adam',
              'adam_lr': 1e-3,
              'num_users': 73515,
              'num_items': 11200,
              'latent_dim': 32,
              'num_negative': 4,
              'layers': [16,64,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
              'l2_regularization': 0.0000001,  # MLP model is sensitive to hyper params
              'use_cuda': True,
              'device_id': 7,
              'pretrain': True,
              'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
              'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'}

neumf_config = {'alias': 'pretrain_neumf_factor8neg4',
                'num_epoch': 200,
                'batch_size': 1024,
                'optimizer': 'adam',
                'adam_lr': 1e-3,
                'num_users': 73515,
                'num_items': 11200,
                'latent_dim_mf': 32,
                'latent_dim_mlp': 32,
                'num_negative': 4,
                'layers': [16,32,16,8],  # layers[0] is the concat of latent user vector & latent item vector
                'l2_regularization': 0.01,
                'use_cuda': True,
                'device_id': 7,
                'pretrain': True,
                'pretrain_mf': 'checkpoints/{}'.format('gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model'),
                'pretrain_mlp': 'checkpoints/{}'.format('mlp_factor8neg4_Epoch100_HR0.5606_NDCG0.2463.model'),
                'model_dir':'checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model'
                }


def main(args):
    sample_generator = SampleGenerator(ratings=pd.read_csv(args.data))
    evaluate_data = sample_generator.evaluate_data

    config = gmf_config
    if args.checkpoints != None:
        config['prev_weights'] = args.checkpoints
        config['load_from_weights'] = True
    else:
        config['load_from_weights'] = False
    config['model_dir'] = args.model_dir
    config['num_epoch'] = args.epochs
    config['batch_size'] = args.batch_size
    config['num_users'] = args.num_users
    config['num_items'] = args.num_items
    config['latent_dim'] = args.latent_dim
    trainer = Trainer(GMF, config)

    for epoch in range(config['num_epoch']):
        print('Epoch {} starts !'.format(epoch))
        print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['batch_size'])
        trainer.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = trainer.evaluate(evaluate_data, epoch_id=epoch)
        trainer.save(config['alias'], epoch, hit_ratio, ndcg)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data',type=str,default='./data/current_user_dataset.csv')
    parser.add_argument('--checkpoints',type=str, default=None)
    parser.add_argument('--model-dir',type=str,required=True,help='The directory where the model will be stored.')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=2048)
    parser.add_argument('--num-users',type=int,default=73517)
    parser.add_argument('--num-items',type=int,default=34520)
    parser.add_argument('--latent-dim',type=int,default=32)

    args = parser.parse_args()

    main(args)
