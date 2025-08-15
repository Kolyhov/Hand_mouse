import threading
import cv2
import numpy as np
import pyautogui

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
        self.visible = False

    def _run(self):
        self.visible = True
        window = 'Hand Mouse Menu'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

        btn_r = 60
        btn_space = 40
        width = btn_space + (btn_r * 2 + btn_space) * 4
        height = 160
        centers = []
        x = btn_space + btn_r
        for _ in range(4):
            centers.append((x, height // 2))
            x += btn_r * 2 + btn_space

        labels = [
            ('Полный\nконтроль', 'full'),
            ('Клавиатура', 'keyboard'),
            ('Спящий', 'sleep'),
            ('Заглушка', 'none'),
        ]

        def draw():
            img = np.full((height, width, 3), 220, dtype=np.uint8)
            buttons = []
            for (text, mode), (cx, cy) in zip(labels, centers):
                cv2.circle(img, (cx, cy), btn_r, (245, 245, 245), -1)
                for i, line in enumerate(text.split('\n')):
                    size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    tx = cx - size[0] // 2
                    ty = cy + btn_r + 20 + i * 20
                    cv2.putText(img, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
                buttons.append((mode, (cx - btn_r, cy - btn_r, cx + btn_r, cy + btn_r)))
            return img, buttons

        img, buttons = draw()

        def on_mouse(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                for mode, (x1, y1, x2, y2) in buttons:
                    if x1 <= mx <= x2 and y1 <= my <= y2:
                        self.selector.set_mode(mode)
                        self.hide()
                        break

        cv2.setMouseCallback(window, on_mouse)
        while self.visible:
            cv2.imshow(window, img)
            if cv2.waitKey(50) == 27:
                break
        cv2.destroyWindow(window)
        self.visible = False

    def show(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def hide(self):
        self.visible = False

    def is_visible(self):
        return self.visible

class KeyboardOverlay:
    def __init__(self):
        self.thread = None
        self.visible = False

    def _run(self):
        self.visible = True
        layout = [
            '1234567890',
            'qwertyuiop',
            'asdfghjkl',
            'zxcvbnm'
        ]
        key_w, key_h, margin = 60, 60, 10
        cols = max(len(r) for r in layout)
        width = margin + (key_w + margin) * cols
        height = margin + (key_h + margin) * len(layout) + key_h + margin

        window = 'Keyboard'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)

        img = np.full((height, width, 3), 220, dtype=np.uint8)
        keys = []
        y = margin
        for row in layout:
            x = margin
            for ch in row:
                rect = (x, y, x + key_w, y + key_h)
                cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (245, 245, 245), -1)
                cv2.putText(img, ch.upper(), (x + 15, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
                keys.append((ch, rect))
                x += key_w + margin
            y += key_h + margin

        exit_rect = (width // 2 - 50, height - key_h - margin, width // 2 + 50, height - margin)
        cv2.rectangle(img, (exit_rect[0], exit_rect[1]), (exit_rect[2], exit_rect[3]), (200, 80, 80), -1)
        cv2.putText(img, 'EXIT', (exit_rect[0] + 10, exit_rect[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        def on_mouse(event, mx, my, flags, param):
            if event == cv2.EVENT_LBUTTONUP:
                for ch, (x1, y1, x2, y2) in keys:
                    if x1 <= mx <= x2 and y1 <= my <= y2:
                        pyautogui.press(ch)
                        return
                x1, y1, x2, y2 = exit_rect
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    self.hide()

        cv2.setMouseCallback(window, on_mouse)
        while self.visible:
            cv2.imshow(window, img)
            if cv2.waitKey(50) == 27:
                break
        cv2.destroyWindow(window)
        self.visible = False

    def show(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def hide(self):
        self.visible = False

    def is_visible(self):
        return self.visible
