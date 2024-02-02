import cv2
import random

class LineSegmentCounter:
    def __init__(self):
        # self.cap = cv2.VideoCapture(video_path)
        # self.cap = cap
        self.segments = []  # Store line segments as (points, color) tuples
        self.points = []
        self.current_segment = None  # Store the current segment as (points, color)
        self.draw = False

        cv2.namedWindow('Video')
        cv2.setMouseCallback('Video', self.mouse_event)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points = [(x, y)]  # Store the starting point
            self.draw = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False
            self.points.append((x, y))  # Store the ending point
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            count = 0
            self.store_segment(self.points, color, count)
        elif self.draw and event == cv2.EVENT_MOUSEMOVE:
            # Update the ending point while dragging the mouse
            self.current_segment = [self.points[0], (x, y)]
            # self.mouse_moves += self.check_mouse_over_segment(x, y)

    def store_segment(self, points, color, count):
        self.segments.append((points, color, count))

    def draw_stored_segments(self, frame, objects_displacement = {}):
        i = 0
        for points, color, count in self.segments:
            if len(points) == 2:
                p1, p2 = points
                frame = cv2.line(frame, p1, p2, color, 4)
                frame, count = self.draw_text_on_segment(frame, p1, p2, count, color, objects_displacement)
            self.segments[i] = (points, color, count)
            i += 1
        
        if self.current_segment and self.draw:
            # Draw the temporary line
            frame = cv2.line(frame, self.current_segment[0], self.current_segment[1], (0, 0, 255), 4)
        return frame

    def draw_text_on_segment(self, frame, p1, p2, count, color, objects_displacement):
        center_x = (p1[0] + p2[0]) // 2
        center_y = (p1[1] + p2[1]) // 2

        # If the object passes throught the line segment then update the count
        for key in objects_displacement.keys():
            [q1, q2] = objects_displacement[key]
            intersection = self.do_intersect(p1, p2, q1, q2)
            if intersection:
                count += 1

        text = f'{count}'        
        frame = cv2.circle(frame, (center_x + 5, center_y - 5), 15, (255,255,255), -1)
        frame = cv2.putText(frame, text, (center_x , center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame , count
    
    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or counterclockwise

    def on_segment(self, p, q, r):
        return (
            (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        )

    def do_intersect(self, p1, q1, p2, q2):
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True  # General case

        if o1 == 0 and self.on_segment(p1, p2, q1):
            return True  # p1, q1, and p2 are collinear and p2 lies on segment p1q1

        if o2 == 0 and self.on_segment(p1, q2, q1):
            return True  # p1, q1, and q2 are collinear and q2 lies on segment p1q1

        if o3 == 0 and self.on_segment(p2, p1, q2):
            return True  # p2, q2, and p1 are collinear and p1 lies on segment p2q2

        if o4 == 0 and self.on_segment(p2, q1, q2):
            return True  # p2, q2, and q1 are collinear and q1 lies on segment p2q2

        return False  # Doesn't fall into any of the above cases

    def run(self, cap):
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            self.draw_stored_segments(frame)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 0
    cap = cv2.VideoCapture(video_path) 
    
    line_segment_drawer = LineSegmentCounter()
    line_segment_drawer.run(cap)
