
from .base_option import BaseOptions

class TestOptions(BaseOptions):
    '''
    add more parameters for training
    '''
    def initialize(self,parser):
        parser = BaseOptions.initialize(self,parser)
        self.isTrain = False
        self.isTest = True
        return parser