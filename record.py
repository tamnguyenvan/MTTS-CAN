import cv2


cap = cv2.VideoCapture(0)

base_size = 72
size = base_size * 3
x1, y1 = (220, 200)
x2, y2 = x1 + size, y1 + size
writer = None
fps = int(cap.get(cv2.CAP_PROP_FPS))
while True:
    ret, frame = cap.read()
    if not ret:
        break

    print('FPS:', fps)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if writer is not None:
        t = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
        cv2.putText(frame, 'Recording. %d' % (t - t0), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255), 2)
        writer.write(frame[y1:y2, x1:x2, :])
    else:
        t0 = int(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

    cv2.imshow('image', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        writer = cv2.VideoWriter('input_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps,  (size, size))

if writer is not None:
    writer.release()
cap.release()