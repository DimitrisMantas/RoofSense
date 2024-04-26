from torchgeo.datamodules import NonGeoDataModule


class TrainingDataModule(NonGeoDataModule):
    def __init__(self,
                 root:str,
                 batch_size: int = 8,
                 num_workers: int|None = None,
                 persistent_workers: bool = True,
                 pin_memory: bool = True,

                 download:bool=False,checksum:bool=False)->None:
        ...