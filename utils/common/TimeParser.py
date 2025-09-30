from datetime import datetime
import os

from entity.common.StrucResult import StrucResult


class TimeParser:

    @classmethod
    def get_current_time(cls):
        return cls.convert_time_format(datetime.now())

    @classmethod
    def convert_time_format(cls, time: datetime):
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def calculate_passing_time(cls, start_time):
        current_time = datetime.now()

        time_interval = (current_time - start_time).seconds / 60

        if time_interval >= float(os.getenv("VALIDATION_REPEAT_SENDING_TIME_INTERVAL")):
            return StrucResult.build_success()
        else:
            return StrucResult.build_error()
