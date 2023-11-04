"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data


def find_dataset_using_name(dataset_name):
    """Import the module "datasets/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, CLS in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(CLS, torch.utils.data.Dataset):
            dataset = CLS

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


# def create_dataset(config):
#     """Create a dataset given the option.

#     This function wraps the class CustomDatasetDataLoader.
#         This is the main interface between this package and 'train.py'/'test.py'

#     Example:
#         >>> from data import create_dataset
#         >>> dataset = create_dataset(opt)
#     """
#     dataset_class = find_dataset_using_name(config['dataset_mode'])
#     dataset = dataset_class(config)
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config['batch_size'],
#         shuffle=not config['serial_batches'],
#         num_workers=int(config['num_threads']))
#     return dataloader

