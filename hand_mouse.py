import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from collections import deque
import time

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

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

gesture_state = "idle"
cooldown      = 0
relative_move = False
last_gesture_time = 0

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

try:
    while cap.isOpened():
        start_time = time.time()
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

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

                h, w, _ = frame.shape
                ix, iy = int(hand.landmark[8].x * w), int(hand.landmark[8].y * h)

                # ===== движение курсора =====
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
                    gesture_state = "move"

                # ===== клики =====
                elif up_cnt == 2 and idx and mid and not scroll_mode and not horn_mode:
                    gesture_state, relative_move = "two_shown", False
                elif up_cnt == 3 and idx and mid and ring and not scroll_mode and not horn_mode:
                    gesture_state, relative_move = "three_shown", False

                # ===== скролл (4 пальца) =====
                elif up_cnt == 4 and idx and mid and ring and pinky and not horn_mode:
                    if not scroll_mode:                 # входим
                        scroll_mode = True
                        scroll_prev_y = iy
                        scroll_trace.clear()
                        scroll_buf = 0.0
                        gesture_state = "scroll"
                    else:                               # скроллим
                        dy_frame = iy - scroll_prev_y   # Δ за кадр
                        scroll_prev_y = iy

                        scroll_trace.append(dy_frame)
                        avg_dy = sum(scroll_trace) / len(scroll_trace)

                        if abs(avg_dy) > SCROLL_DEADZONE_PX:
                            scroll_buf += avg_dy / 100 * SCROLL_SENSITIVITY
                            steps = int(scroll_buf)
                            if steps:
                                pyautogui.scroll(steps)
                                scroll_buf -= steps  # оставляем дробную часть

                # ===== «рога» (index + pinky) → Ctrl+← / Ctrl+→ =====
                # допускаем небольшие ошибки распознавания кольца,
                # поэтому не проверяем up_cnt, а требуем лишь указательный
                # и мизинец при опущенных большом и среднем пальцах
                elif idx and pinky and not thumb and not mid and not scroll_mode:
                    if not horn_mode:                  # включаем режим
                        horn_mode = True
                        horn_ref_x = ix
                    else:                              # смотрим смещение
                        dx = ix - horn_ref_x
                        if abs(dx) > HORN_DEADZONE_PX and cooldown == 0:
                            if dx < 0:
                                pyautogui.hotkey('ctrl', 'right')
                            else:
                                pyautogui.hotkey('ctrl', 'left')
                            cooldown, horn_mode = 10, False  # ждём и выходим

                else:
                    # выходим из скролла
                    if scroll_mode and not (up_cnt == 4 and idx and mid and ring and pinky):
                        scroll_mode = False
                        gesture_state = "idle"
                    # выходим из «рогов»
                    if horn_mode and not (idx and pinky and not thumb and not mid):
                        horn_mode = False

                    # клики по отпусканию
                    if gesture_state == "two_shown" and not (idx or mid) and cooldown == 0:
                        pyautogui.click()
                        cooldown, gesture_state = 10, "idle"
                    elif gesture_state == "three_shown" and not (idx or mid or ring) and cooldown == 0:
                        pyautogui.click(button='right')
                        cooldown, gesture_state = 10, "idle"
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

        # кулдаун
        if cooldown:
            cooldown -= 1
        # потеряли руку — сбросы
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

        # Выход по нажатию 'q' вместо ESC
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if gesture_state != "idle":
            last_gesture_time = time.time()

        # Динамическое ограничение частоты кадров
        active_gesture = gesture_state != "idle" or (time.time() - last_gesture_time < 1.0)
        target_fps = FPS_GESTURE if active_gesture else FPS_IDLE
        elapsed = time.time() - start_time
        sleep_time = max(0, 1.0 / target_fps - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("Программа остановлена пользователем")

finally:
    cap.release()
    cv2.destroyAllWindows()
