import cv2
from realsense_wrapper import RealsenseWrapper
rw = RealsenseWrapper()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("video.mp4", fourcc, 20.0, (640,480))

images = []
ok = True
while ok:
    frame = rw.get_color_frame()
    if frame is not None:
        out.write(frame)
        cv2.imshow("im", frame)
        if cv2.waitKey(1) == 13:
            ok = False

out.release()


    

