import cv2

def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    i = 0
    # a variable to set how many frames you want to skip
    frame_skip = 8
    # a variable to keep track of the frame to be saved
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            cv2.imwrite("images/"+f"{frame_count:04d}.jpg", frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()