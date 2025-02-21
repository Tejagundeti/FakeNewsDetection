import cv2  # Import OpenCV at the top

# Now the rest of the code
video_path = "deepfake_video.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(f"frames/frame_{frame_count}.jpg", frame)
    frame_count += 1

cap.release()