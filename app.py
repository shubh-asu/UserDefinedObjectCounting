from draw_count import LineSegmentCounter
from oop_detection import ObjectDetector
import cv2

# cap = cv2.VideoCapture(0)  
cap = cv2.VideoCapture('http://192.168.0.204:8000/stream.mjpg') 
UserDefinedCounter = LineSegmentCounter()
ObjectRecognizer = ObjectDetector()

output = cv2.VideoWriter("./ProjectVideo1.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 5, (640, 480) )

prev_object_centroid_dict = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    result_frame, centers = ObjectRecognizer.detect_objects(frame)

    # Update the centroid tracker with the new set of object centers
    objects = ObjectRecognizer.centroid_tracker.update(centers)

    curr_object_centroid_dict = {}
    # Draw the bounding boxes and centroids
    for (object_id, centroid) in objects.items():
        text = f"ID {object_id}"
        cv2.putText(result_frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.circle(result_frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        curr_object_centroid_dict[object_id] = [centroid[0], centroid[1]]

    # Get current object centroids and previous object centroids
    common_keys = set(curr_object_centroid_dict.keys()) & set(prev_object_centroid_dict.keys())
    object_displacement = {key: [curr_object_centroid_dict[key], prev_object_centroid_dict[key]] for key in common_keys}

    object_detection_result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
    
    # Pass displacement of objects to the Counters for predicting if the object went through the line segment or not
    final_frame = UserDefinedCounter.draw_stored_segments(object_detection_result_frame_rgb, object_displacement)
    cv2.imshow('Video', final_frame)
    output.write(final_frame)

    prev_object_centroid_dict = curr_object_centroid_dict

    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break

cap.release()
output.release()
cv2.destroyAllWindows()