import cv2
import mediapipe as mp
import numpy as np

# Initialize the MediaPipe Hands tracker
mp_hands = mp.solutions.hands

# Initialize variables for storing previous fingertip position
prev_x, prev_y = None, None

# Create a whiteboard as a black canvas
whiteboard = np.zeros((480, 640, 4), dtype=np.uint8)
whiteboard[:, :, 3] = 255

# Start the webcam
cap = cv2.VideoCapture(0)

idle = False

# Loop until the user presses the "q" key
while True:

    pointer = np.zeros((480, 640, 4), dtype=np.uint8)
    
    # Capture the next frame from the webcam
    ret, frame = cap.read()
    x, y = 0, 0

    # If the frame is empty, break the loop
    if not ret:
        break

    # Convert the frame to RGB
    frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)

    # Process the frame with the MediaPipe Hands tracker
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        results = hands.process(frame)

    # If a hand is detected
    if results.multi_hand_landmarks:
        # Get the hand landmark for the index fingertip
        index_fingertip = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Convert the fingertip position to pixel coordinates
        x, y = int(index_fingertip.x * 640), int(index_fingertip.y * 480)

        cv2.circle(pointer, (x, y), 2, (255, 255, 255, 255), 5)

        if not idle:
            # If it's the first frame or previous coordinates are None, initialize them
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y

            # Draw a line from the previous position to the current position on the whiteboard
            cv2.line(whiteboard, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

            # Update the previous position
            prev_x, prev_y = x, y

    # Display the combined image of the webcam frame and the whiteboard
    cv2.imshow("Whiteboard", cv2.addWeighted(whiteboard, 1, pointer, 1, 0))

    # Get the user's input
    key = cv2.waitKey(1)
    
    if key == ord('z'):
        idle = True
        prev_x, prev_y = None, None
    else:
        idle = False

    # If the user presses the "q" key, break the loop
    if key == ord("q"):
        break
    elif key == ord('x'):
        whiteboard = np.zeros((480, 640, 4), dtype=np.uint8)
        whiteboard[:, :, 3] = 255
    
# Release the webcam
cap.release()
cv2.destroyAllWindows()
