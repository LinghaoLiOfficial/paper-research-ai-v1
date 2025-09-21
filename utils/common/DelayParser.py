import time
import random


# 模拟用户行为的延迟
class DelayParser:

    @classmethod
    def execute_delay(cls, need):
        delay = random.uniform(need, need + 2)
        time.sleep(delay)

    @classmethod
    def execute_short_delay(cls):
        delay = random.uniform(2, 3)
        time.sleep(delay)

    @classmethod
    def execute_medium_delay(cls):
        delay = random.uniform(5, 7)
        time.sleep(delay)

    @classmethod
    def execute_long_delay(cls):
        delay = random.uniform(10, 13)
        time.sleep(delay)

