class BaseName:

    @classmethod
    def to_list(cls):
        return [v for k, v in cls.__dict__.items() if (not (k.startswith("__") and k.endswith("__")))]
