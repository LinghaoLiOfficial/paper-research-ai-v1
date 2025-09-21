from buffer.CommonBuffer import CommonBuffer


class GlobalInit(CommonBuffer):
    _has_init = False

    @classmethod
    def check(cls):
        return cls._has_init

    @classmethod
    def update(cls):
        with cls._lock:
            cls._has_init = True
