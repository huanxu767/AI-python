import cv2
camera = cv2.VideoCapture(source)
while True:
    (grabbed, frame) = camera.read()
    while not grabbed:
        continue
    cv2.imshow("capture", frame)
    # cv2.waitKey(0)  # 捕获并显示一帧，按键后捕获并显示新的一帧
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#%%
