import warnings
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class DatasetTemplate(LightningDataModule):
    def __init__(self, *args, **kwargs):
        '''
            Initialize your variables
            Note: Specify your arguments' datatype for a comprehensive code
        '''
        super().__init__()
        warnings.warn("You're not overriding DatasetTemplate.__init__ method")        


    def prepare_data(self):
        '''
            In case we need to download or tokenize sth. More details at https://lightning.ai/docs/pytorch/latest/data/datamodule.html#prepare-data
        '''
        warnings.warn("You're not overriding DatasetTemplate.prepare_data method")        


    def setup(self, stage: str):
        '''
            Set up your internal processing to initialize dataset
            For example: Initalize your torch.utils.data.Dataset
            Note: setup() is called after prepare_data()
        '''
        warnings.warn("You're not overriding DatasetTemplate.setup method")        


    def train_loader(self) -> DataLoader:
        '''
            Return a Dataloader
        '''
        raise NotImplementedError('DatasetTemplate.train_loader must return a torch.utils.data.DataLoader')


    def val_loader(self) -> DataLoader:
        '''
            Return a Dataloader
        '''
        raise NotImplementedError('DatasetTemplate.val_loader must return a torch.utils.data.DataLoader')


    def test_loader(self) -> DataLoader:
        '''
            Return a Dataloader
        '''
        raise NotImplementedError('DatasetTemplate.test_loader must return a torch.utils.data.DataLoader')


    def predict_loader(self) -> DataLoader:
        '''
            Return a Dataloader
        '''
        raise NotImplementedError('DatasetTemplate.predict_loader must return a torch.utils.data.DataLoader')


    def teardown(self, stage: str):
        '''
            Used to clean up when run is finished
        '''
        warnings.warn("You're not overriding DatasetTemplate.teardown method")        
