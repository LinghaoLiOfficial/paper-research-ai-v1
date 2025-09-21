import torch


class DeviceParser:

    @classmethod
    def load_device(cls):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return device
