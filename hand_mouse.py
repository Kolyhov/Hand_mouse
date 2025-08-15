import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque
import time

# ==== режимы работы ====
MODE_SLEEP = "sleep"
MODE_MENU = "menu"
MODE_FULL = "full"
MODE_KEYBOARD = "keyboard"

SCREEN_W, SCREEN_H = pyautogui.size()

# — настройки —
SENSITIVITY          = 6.0   # курсор
SMOOTHING_WINDOW     = 14
SCROLL_SENSITIVITY   = 24     # сколько тиков ≈ 100 px
SCROLL_AVG_FRAMES    = 10     # сколько последних Δy усредняем
SCROLL_DEADZONE_PX   = 2     # игнорируем дрожь < 2 px
# меньше свободный ход — жест «рога» срабатывает быстрее и надёжнее
HORN_DEADZONE_PX     = 5    # «свободный ход» для жеста index+pinky
PALM_Z_DIFF_THRESH   = 0.2   # max |z_5 - z_17|, чтобы ладонь была повернута к камере
MASK_FRAMES         = 3     # жест подтверждается после N одинаковых кадров

# Частоты кадров: базовая и при активном жесте
FPS_IDLE             = 5
FPS_GESTURE          = 30
FPS_SLEEP            = 1

INACTIVITY_TIMEOUT   = 15 * 60  # 15 минут бездействия выключают полный режим

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesture_state = "idle"
cooldown      = 0
relative_move = False

mode               = MODE_SLEEP
last_gesture_time  = time.time()
ten_fingers_frames = 0

menu_buttons    = []
keyboard_keys   = []
keyboard_exit   = None

scroll_mode   = False
scroll_prev_y = 0
scroll_buf    = 0.0
scroll_trace  = deque(maxlen=SCROLL_AVG_FRAMES)

# ==== новый жест (index + pinky) ====
horn_mode   = False          # активный ли «рог»-режим
horn_ref_x  = 0              # точка, откуда считаем смещение

ref_x = ref_y = 0
trace = deque(maxlen=SMOOTHING_WINDOW)
finger_history = deque(maxlen=MASK_FRAMES)
confirmed_mask = (False, False, False, False, False)

def fingers_up(land, angle_thresh=160):
    """Возвращает, какие пальцы выпрямлены.

    Проверка выполняется по углам в MCP и PIP, что устойчиво к наклону
    кисти относительно камеры.
    """

    def ang(a, b, c):
        v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    joints = [
        (2, 3, 4),   # thumb
        (5, 6, 7),   # index
        (9, 10, 11), # middle
        (13, 14, 15),# ring
        (17, 18, 19) # pinky
    ]

    out = []
    for mcp, pip, dip in joints:
        mcp_ang = ang(land.landmark[0], land.landmark[mcp], land.landmark[pip])
        pip_ang = ang(land.landmark[mcp], land.landmark[pip], land.landmark[dip])
        out.append(mcp_ang > angle_thresh and pip_ang > angle_thresh)

    return tuple(out)  # (thumb, index, middle, ring, pinky)


def draw_menu(frame):
    global menu_buttons
    h, w, _ = frame.shape
    overlay = frame.copy()
    box_w, box_h = 400, 200
    x1, y1 = w // 2 - box_w // 2, h // 2 - box_h // 2
    x2, y2 = x1 + box_w, y1 + box_h
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    labels = [
        ("full", "M"),
        ("keyboard", "K"),
        ("sleep", "Z"),
        ("dummy", "?")
    ]
    menu_buttons = []
    radius = 40
    spacing = box_w // 4
    for i, (name, icon) in enumerate(labels):
        cx = x1 + spacing // 2 + i * spacing + spacing // 2
        cy = (y1 + y2) // 2 - 20
        cv2.circle(frame, (cx, cy), radius, (200, 200, 200), -1)
        cv2.putText(frame, icon, (cx - 15, cy + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
        cv2.putText(frame, name, (cx - 50, cy + radius + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        menu_buttons.append({"name": name, "center": (cx, cy), "radius": radius})


def draw_keyboard(frame):
    global keyboard_keys, keyboard_exit
    h, w, _ = frame.shape
    overlay = frame.copy()
    kb_w, kb_h = int(w * 0.8), int(h * 0.4)
    x1, y1 = w // 2 - kb_w // 2, h // 2 - kb_h // 2
    x2, y2 = x1 + kb_w, y1 + kb_h
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
    frame[:] = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    rows = ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"]
    key_w = kb_w // 10
    key_h = kb_h // 4
    keyboard_keys = []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            kx1 = x1 + c * key_w
            ky1 = y1 + r * key_h
            kx2 = kx1 + key_w
            ky2 = ky1 + key_h
            cv2.rectangle(frame, (kx1 + 5, ky1 + 5), (kx2 - 5, ky2 - 5), (200, 200, 200), -1)
            cv2.putText(frame, ch, (kx1 + 20, ky1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
            keyboard_keys.append({"char": ch.lower(), "box": (kx1 + 5, ky1 + 5, kx2 - 5, ky2 - 5)})

    # space bar
    sp_x1 = x1 + key_w
    sp_x2 = x1 + kb_w - key_w
    sp_y1 = y2 - key_h
    sp_y2 = sp_y1 + key_h - 5
    cv2.rectangle(frame, (sp_x1, sp_y1), (sp_x2, sp_y2), (200, 200, 200), -1)
    cv2.putText(frame, "space", (sp_x1 + 20, sp_y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 50), 2)
    keyboard_keys.append({"char": "space", "box": (sp_x1, sp_y1, sp_x2, sp_y2)})

    # exit button
    ex_size = 30
    ex_x1 = x2 - ex_size - 10
    ex_y1 = y1 + 10
    ex_x2 = ex_x1 + ex_size
    ex_y2 = ex_y1 + ex_size
    cv2.rectangle(frame, (ex_x1, ex_y1), (ex_x2, ex_y2), (200, 200, 200), -1)
    cv2.putText(frame, "X", (ex_x1 + 7, ex_y1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
    keyboard_exit = (ex_x1, ex_y1, ex_x2, ex_y2)


def handle_menu_click(ix, iy):
    for btn in menu_buttons:
        cx, cy = btn["center"]
        if (ix - cx) ** 2 + (iy - cy) ** 2 <= btn["radius"] ** 2:
            name = btn["name"]
            if name == "full":
                return MODE_FULL
            if name == "keyboard":
                return MODE_KEYBOARD
            if name == "sleep":
                return MODE_SLEEP
            return MODE_MENU
    return MODE_MENU


def handle_keyboard_click(ix, iy):
    x1, y1, x2, y2 = keyboard_exit
    if x1 <= ix <= x2 and y1 <= iy <= y2:
        return "exit"
    for key in keyboard_keys:
        kx1, ky1, kx2, ky2 = key["box"]
        if kx1 <= ix <= kx2 and ky1 <= iy <= ky2:
            pyautogui.press(key["char"])
            return key["char"]
    return None

try:
    while cap.isOpened():
        start_time = time.time()
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        h, w, _ = frame.shape

        if mode == MODE_FULL and time.time() - last_gesture_time > INACTIVITY_TIMEOUT:
            mode = MODE_MENU

        # обнаружение 10 пальцев
        if res.multi_hand_landmarks and len(res.multi_hand_landmarks) >= 2:
            all_up = all(sum(fingers_up(hand)) == 5 for hand in res.multi_hand_landmarks[:2])
            if all_up:
                ten_fingers_frames += 1
            else:
                ten_fingers_frames = 0
        else:
            ten_fingers_frames = 0
        if ten_fingers_frames >= MASK_FRAMES:
            mode = MODE_MENU
            gesture_state = "idle"
            scroll_mode = False
            horn_mode = False
            relative_move = False
            trace.clear()
            scroll_trace.clear()
            ten_fingers_frames = 0

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            palm_ok = abs(hand.landmark[5].z - hand.landmark[17].z) < PALM_Z_DIFF_THRESH

            if palm_ok:
                current_mask = fingers_up(hand)
                finger_history.append(current_mask)
                if len(finger_history) == MASK_FRAMES and all(m == current_mask for m in finger_history):
                    confirmed_mask = current_mask
                thumb, idx, mid, ring, pinky = confirmed_mask
                up_cnt = sum(confirmed_mask)

                ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)

                if mode != MODE_SLEEP:
                    if up_cnt == 1 and idx and not scroll_mode and not horn_mode:
                        if not relative_move:
                            ref_x, ref_y = ix, iy
                            relative_move = True
                            trace.clear()
                        dx = (ix - ref_x) * SENSITIVITY
                        dy = (iy - ref_y) * SENSITIVITY
                        ref_x, ref_y = ix, iy
                        trace.append((dx, dy))
                        avg_dx = sum(x for x, _ in trace) / len(trace)
                        avg_dy = sum(y for _, y in trace) / len(trace)
                        if abs(avg_dx) > .5 or abs(avg_dy) > .5:
                            pyautogui.moveRel(avg_dx, avg_dy, _pause=False)
                            last_gesture_time = time.time()
                        gesture_state = "move"

                    elif up_cnt == 2 and idx and mid and not scroll_mode and not horn_mode:
                        gesture_state, relative_move = "two_shown", False
                    elif up_cnt == 3 and idx and mid and ring and not scroll_mode and not horn_mode and mode == MODE_FULL:
                        gesture_state, relative_move = "three_shown", False
                    elif up_cnt == 4 and idx and mid and ring and pinky and not horn_mode and mode == MODE_FULL:
                        if not scroll_mode:
                            scroll_mode = True
                            scroll_prev_y = iy
                            scroll_trace.clear()
                            scroll_buf = 0.0
                            gesture_state = "scroll"
                        else:
                            dy_frame = iy - scroll_prev_y
                            scroll_prev_y = iy
                            scroll_trace.append(dy_frame)
                            avg_dy = sum(scroll_trace) / len(scroll_trace)
                            if abs(avg_dy) > SCROLL_DEADZONE_PX:
                                scroll_buf += avg_dy / 100 * SCROLL_SENSITIVITY
                                steps = int(scroll_buf)
                                if steps:
                                    pyautogui.scroll(steps)
                                    scroll_buf -= steps
                                    last_gesture_time = time.time()
                    elif idx and pinky and not thumb and not mid and not scroll_mode and mode == MODE_FULL:
                        if not horn_mode:
                            horn_mode = True
                            horn_ref_x = ix
                        else:
                            dx = ix - horn_ref_x
                            if abs(dx) > HORN_DEADZONE_PX and cooldown == 0:
                                if dx < 0:
                                    pyautogui.hotkey('ctrl', 'right')
                                else:
                                    pyautogui.hotkey('ctrl', 'left')
                                cooldown, horn_mode = 10, False
                                last_gesture_time = time.time()
                    else:
                        if scroll_mode and not (up_cnt == 4 and idx and mid and ring and pinky):
                            scroll_mode = False
                            gesture_state = "idle"
                        if horn_mode and not (idx and pinky and not thumb and not mid):
                            horn_mode = False

                        if gesture_state == "two_shown" and not (idx or mid) and cooldown == 0:
                            if mode == MODE_MENU:
                                mode = handle_menu_click(ix, iy)
                            elif mode == MODE_KEYBOARD:
                                res_kb = handle_keyboard_click(ix, iy)
                                if res_kb == "exit":
                                    mode = MODE_MENU
                            else:
                                pyautogui.click()
                                last_gesture_time = time.time()
                            cooldown, gesture_state = 10, "idle"
                        elif gesture_state == "three_shown" and not (idx or mid or ring) and cooldown == 0 and mode == MODE_FULL:
                            pyautogui.click(button='right')
                            cooldown, gesture_state = 10, "idle"
                            last_gesture_time = time.time()

            else:
                relative_move = False
                if scroll_mode:
                    scroll_mode = False
                if horn_mode:
                    horn_mode = False
                trace.clear()
                scroll_trace.clear()
                scroll_buf = 0.0
                finger_history.clear()
                confirmed_mask = (False, False, False, False, False)
                gesture_state = "idle"

        if mode == MODE_MENU:
            draw_menu(frame)
        elif mode == MODE_KEYBOARD:
            draw_keyboard(frame)

        if cooldown:
            cooldown -= 1
        if not res.multi_hand_landmarks:
            relative_move = False
            if scroll_mode:
                scroll_mode = False
            if horn_mode:
                horn_mode = False
            trace.clear()
            scroll_trace.clear()
            scroll_buf = 0.0
            finger_history.clear()
            confirmed_mask = (False, False, False, False, False)
            gesture_state = "idle"

        cv2.imshow('HandMouse', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if mode == MODE_SLEEP:
            target_fps = FPS_SLEEP
        else:
            target_fps = FPS_GESTURE if gesture_state != "idle" else FPS_IDLE
        elapsed = time.time() - start_time
        sleep_time = max(0, 1.0 / target_fps - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Программа остановлена пользователем")

finally:
    cap.release()
    cv2.destroyAllWindows()
