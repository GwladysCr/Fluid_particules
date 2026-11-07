# Camera & hand detection test
def open_camera(index=CAM_INDEX, w=FRAME_W, h=FRAME_H):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la caméra {index}")
    return cap


class Hand:
    # Petit conteneur : 21 points (x, y, z normalisés)
    def __init__(self, landmarks):
        self.landmark = landmarks


def create_landmarker(num_hands=2):
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(options)


_last_ts = 0
def detect(landmarker, frame_bgr):
    # Timestamps (ms) strictement croissants
    global _last_ts
    ts = max(int(time.monotonic() * 1000), _last_ts + 1)
    _last_ts = ts
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_img, ts)


def get_target_hand(results, label=TARGET_HAND):
    # effet miroir avant mp to match label
    for lms, handed in zip(results.hand_landmarks, results.handedness):
        if handed[0].category_name == label:
            return Hand(lms)
    return None


def draw_hand(img, lms, color=(0, 255, 120)):
    h, w = img.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lms]
    for conn in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(img, pts[conn.start], pts[conn.end], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 4, (255, 255, 255), -1, cv2.LINE_AA)


FINGER_TIPS = {"Pouce": (4, 3), "Index": (8, 6), "Majeur": (12, 10), "Annulaire": (16, 14), "Auriculaire": (20, 18)}


def fingers_state(hand_lms, w, h):
    # Doigt tendu = bout plus loin du poignet que l'articulation PIP
    pts = np.array([[l.x * w, l.y * h] for l in hand_lms.landmark])
    d = lambda a, b: np.linalg.norm(pts[a] - pts[b])
    state = {"Pouce": d(4, 17) > d(3, 17) * 1.1}
    for name, (tip, pip) in list(FINGER_TIPS.items())[1:]:
        state[name] = d(tip, 0) > d(pip, 0) * 1.05
    pinch = d(4, 8) / max(d(0, 9), 1e-6)   # distance pouce-index normalisée
    return state, pinch


def run_camera_test(max_seconds=None):
    cap = open_camera()
    t0 = t_prev = time.time()
    fps = 0.0
    try:
        with create_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Lecture caméra impossible")
                    break
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                res = detect(landmarker, frame)

                for hand_lms in res.hand_landmarks:
                    draw_hand(frame, hand_lms)

                target = get_target_hand(res)
                if target is not None:
                    state, pinch = fingers_state(target, FRAME_W, FRAME_H)
                    n_up = sum(state.values())
                    cv2.putText(frame, f"Main {TARGET_HAND}: {n_up} doigt(s) leve(s)", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    for i, (name, up) in enumerate(state.items()):
                        col = (0, 255, 0) if up else (80, 80, 255)
                        cv2.putText(frame, f"{name}: {'OUI' if up else 'non'}", (15, 70 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
                    cv2.putText(frame, f"Pincement: {pinch:.2f}", (15, 70 + 28 * 5 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, f"Main {TARGET_HAND} non detectee", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                now = time.time()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                cv2.putText(frame, f"{fps:.0f} FPS - q pour quitter", (FRAME_W - 280, FRAME_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Test camera + doigts", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if max_seconds and now - t0 > max_seconds:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1) 

# ESC to quit
run_camera_test()
