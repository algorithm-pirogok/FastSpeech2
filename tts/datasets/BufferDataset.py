from tts.base.base_dataset import BaseDataset


class BufferDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
