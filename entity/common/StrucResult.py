class StrucResult:
    check: bool
    code: int
    message: str
    data: dict

    def __init__(self, check, code, message, data):
        self.check = check
        self.code = code
        self.message = message
        self.data = data

    def to_dict(self):
        return self.__dict__

    def get_data(self, key):
        return self.data.get(key)

    def get_data_on_results(self):
        return self.get_data("results")

    def verify_data(self, key: str):
        value = self.get_data(key)
        if value == []:
            return False

        return True

    def verify_data_on_results(self):
        return self.verify_data("results")

    @classmethod
    def _build(cls, check: bool, code: int, message: str, data: dict):
        return cls(check, code, message, data)

    @classmethod
    def build_success(cls, code: int = None, message: str = "", data: dict = {}):
        return cls._build(
            check=True,
            code=code,
            message=message,
            data=data
        )

    @classmethod
    def build_success_with_results(cls, value):
        return cls.build_success(
            data={
                "results": value
            }
        )

    @classmethod
    def build_error(cls, code: int = None, message: str = "", data: dict = {}):
        return cls._build(
            check=False,
            code=code,
            message=message,
            data=data
        )

