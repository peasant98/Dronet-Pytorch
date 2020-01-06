import torch
from dronet_datasets import DronetDataset


def testModel(model: torch.nn.Module, weights_path=None):
    '''
    tests the model
    '''
    # go through dataset, run the trained (hopefully) model from path
    if weights_path != None:
        try:
            model.load_state_dict(torch.load(weights_path))
        except Exception as e:
            print('Could not fill up weights,', e)
    
    testing_dataset = DronetDataset('all-data', 'testing', True)
    model.eval()

    testing_dataloader = torch.utils.data.DataLoader(testing_dataset, batch_size=2, 
                                            shuffle=True, num_workers=2)
    # evaluate model
    # use rmse and eva for metrics