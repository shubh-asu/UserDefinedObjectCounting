import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import cv2
import numpy as np

class ObjectDetector:
    def __init__(self, target_class=1, confidence_threshold=0.7):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
        self.transform = T.Compose([T.ToTensor()])
        self.target_class = target_class
        self.confidence_threshold = confidence_threshold
        self.centroid_tracker = CentroidTracker()

    def detect_objects(self, frame):
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_image = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_image)

        result_frame = np.array(pil_image)
        centers = []

        for box, label, score in zip(
            prediction[0]['boxes'].cpu().numpy(),
            prediction[0]['labels'].cpu().numpy(),
            prediction[0]['scores'].cpu().numpy()
        ):
            if label == self.target_class and score > self.confidence_threshold:
                box = list(map(int, box))
                center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
                centers.append(center)

                color = (0, 255, 0)  # Green color for bounding boxes
                result_frame = cv2.rectangle(result_frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                result_frame = cv2.putText(
                    result_frame,
                    f"Class: {label}, Score: {score:.2f}",
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
                result_frame = cv2.circle(result_frame, center, 3, color, -1)

        return result_frame, centers

    def run_detection(self):
        cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change to another value if using an external camera

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            result_frame, centers = self.detect_objects(frame)

            # Update the centroid tracker with the new set of object centers
            objects = self.centroid_tracker.update(centers)

            # Draw the bounding boxes and centroids
            for (object_id, centroid) in objects.items():
                text = f"ID {object_id}"
                cv2.putText(result_frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.circle(result_frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

            result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('Object Detection with Tracking', result_frame_rgb)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.objects = {}
        self.next_object_id = 1
        self.max_disappeared = max_disappeared

    def update(self, centers):
        objects = {}

        for center in centers:
            object_id = self.register_object(center)
            objects[object_id] = center

        self.cleanup_unused_objects(objects)

        return objects

    def register_object(self, center):
        # Check if there are existing objects
        if not self.objects:
            object_id = self.next_object_id
            self.objects[object_id] = {'center': center, 'disappeared': 0}
            self.next_object_id += 1
            return object_id

        # Calculate Euclidean distance to find the closest object
        object_ids = list(self.objects.keys())
        object_data = list(self.objects.values())
        object_centers = [data['center'] for data in object_data]
        distances = [np.linalg.norm(np.array(center) - np.array(obj_center)) for obj_center in object_centers]
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]

        # Register a new object if the closest existing object is too far away
        if min_distance > 120:  # Adjust this threshold as needed
            object_id = self.next_object_id
            self.objects[object_id] = {'center': center, 'disappeared': 0}
            self.next_object_id += 1
            return object_id
        else:
            # Update the closest existing object
            object_id = object_ids[min_distance_index]
            self.objects[object_id] = {'center': center, 'disappeared': 0}
            return object_id

    def cleanup_unused_objects(self, current_objects):
        # Create a copy of object IDs to avoid dictionary size change during iteration
        object_ids_to_remove = []
        for object_id, object_data in self.objects.items():
            if object_id not in current_objects:
                object_data['disappeared'] += 1

                # Remove the object if it has disappeared for too many frames
                if object_data['disappeared'] > self.max_disappeared:
                    object_ids_to_remove.append(object_id)

        # Remove the objects outside the iteration to avoid dictionary size change
        for object_id in object_ids_to_remove:
            del self.objects[object_id]

        # Reset disappeared counter for objects that are still present
        for object_id in current_objects:
            if object_id in self.objects:
                self.objects[object_id]['disappeared'] = 0


if __name__ == "__main__":
    detector = ObjectDetector()
    detector.run_detection()
