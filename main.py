import cv2
import mediapipe
import time

ctime = 0
ptime = 0

cap = cv2.VideoCapture(0)

medhands = mediapipe.solutions.hands
hands = medhands.Hands(max_num_hands=1, min_detection_confidence=0.7)
draw = mediapipe.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(imgrgb)

    lmlist = []
    tipids = [4, 8, 12, 16, 20]

    if res.multi_hand_landmarks:
        for handlms in res.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):

                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                draw.draw_landmarks(img, handlms, medhands.HAND_CONNECTIONS,
                                    draw.DrawingSpec(color=(0, 255, 204), thickness=2, circle_radius=2),
                                    draw.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=3))

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, f'FPS:{str(int(fps))}', (0, 12), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    cv2.imshow("hand gestures", img)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
