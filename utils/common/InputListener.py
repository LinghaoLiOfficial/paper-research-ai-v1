import threading
import time
import keyboard


class InputListener:

    is_running = False
    is_paused = False
    keyboard_thread = None

    @classmethod
    def listen_for_keyboard(cls):
        print("[后台键盘监听程序已开启]")
        while cls.is_running:
            if keyboard.is_pressed("alt+p"):
                print("[状态更变为暂停状态]")
                cls.is_paused = True
                while cls.is_paused:
                    if keyboard.is_pressed("alt+b"):
                        print("[状态变更为运行状态]")
                        cls.is_paused = False
                        break
                    time.sleep(0.1)
            time.sleep(0.1)
        print("[后台键盘监听程序已终止]")

    # 启动键盘监听
    # daemon=True 意味着启动的是守护线程，不会阻塞主线程
    @classmethod
    def activate(cls):
        cls.is_running = True
        cls.keyboard_thread = threading.Thread(target=cls.listen_for_keyboard, daemon=True)
        cls.keyboard_thread.start()

    @classmethod
    def deactivate(cls):
        cls.is_running = False

    @classmethod
    def check(cls):
        if not cls.is_running:
            return False

        return cls.is_paused

    @classmethod
    def listen(cls):
        if cls.check():
            print("[程序暂停运行]")
            while cls.is_paused:
                if not cls.is_running:
                    break
                time.sleep(0.2)
            print("[程序继续运行]")

