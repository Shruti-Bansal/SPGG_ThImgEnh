'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt, phase):
    '''create dataset'''
    mode = dataset_opt['mode']
    #if mode == 'LR':
    #    from data.LR_dataset import LRDataset as D
    #elif mode == 'LRHR':
    #    from data.LRHR_dataset import LRHRDataset as D
    #else:
    #    raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    from data.MS2_Dataset import DataLoader_MS2 as D
    #dataset = D(dataset_opt)
    if phase == 'train':
        datalist_name = "/ocean/projects/cis220039p/bansals/SPGG_ThImgEnh/code/filenames/ms2_train.txt"
    else:
        datalist_name = "/ocean/projects/cis220039p/bansals/SPGG_ThImgEnh/code/filenames/ms2_test.txt"
    
    dataset = D(
    #root="/ocean/projects/cis220039p/shared/datasets/MS2_full/",
    root="/ocean/projects/cis220039p/bansals/Data/MS2/",
    datalist=datalist_name,
    data_split=phase,
    process="minmax",
    resolution="640x256", #check resolution
    sampling_step=1,
    set_length=1,
    set_interval=1,
    )
    #logger = logging.getLogger('base')
    #logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
    #                                                       dataset_opt['name']))
    return dataset
