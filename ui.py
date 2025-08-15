import threading
import tkinter as tk
from tkinter import ttk

# Shared state variable for selected mode
class ModeSelector:
    def __init__(self):
        self.mode = None
        self.lock = threading.Lock()

    def set_mode(self, mode):
        with self.lock:
            self.mode = mode

    def get_mode(self):
        with self.lock:
            return self.mode

class Overlay:
    def __init__(self, selector: ModeSelector):
        self.selector = selector
        self.thread = None
        self.root = None

    def _run(self):
        self.root = tk.Tk()
        self.root.title('Hand Mouse')
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)
        self.root.configure(bg='#80FFFFFF')  # semi-transparent

        frame = ttk.Frame(self.root, padding=20)
        frame.pack()

        def add_btn(text, mode):
            btn = ttk.Button(frame, text=text, command=lambda: self._choose(mode))
            btn.pack(side=tk.LEFT, padx=10)

        add_btn('Полный\nконтроль', 'full')
        add_btn('Клавиатура', 'keyboard')
        add_btn('Спящий', 'sleep')
        add_btn('Заглушка', 'none')

        self.root.mainloop()

    def _choose(self, mode):
        self.selector.set_mode(mode)
        self.hide()

    def show(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def hide(self):
        if self.root:
            self.root.quit()
            self.root = None

class KeyboardOverlay:
    def __init__(self):
        self.thread = None
        self.root = None

    def _run(self):
        self.root = tk.Tk()
        self.root.title('Keyboard')
        self.root.attributes('-topmost', True)
        self.root.configure(bg='#80FFFFFF')

        frame = ttk.Frame(self.root, padding=10)
        frame.pack()

        keys = [
            '1234567890',
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]

        import pyautogui

        for row in keys:
            row_frame = ttk.Frame(frame)
            row_frame.pack()
            for ch in row:
                ttk.Button(row_frame, text=ch.upper(), width=4,
                           command=lambda c=ch: pyautogui.press(c)).pack(side=tk.LEFT, padx=2, pady=2)

        ttk.Button(frame, text='Выход', command=self.hide).pack(pady=10)

        self.root.mainloop()

    def show(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def hide(self):
        if self.root:
            self.root.quit()
            self.root = None
