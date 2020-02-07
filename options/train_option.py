import os
from .base_option import BaseOptions

class TrainOptions(BaseOptions):
    '''
    add more parameters for training
    '''
    def initialize(self,parser):
        parser = BaseOptions.initialize(self,parser)
        self.isTrain = True
        return parser