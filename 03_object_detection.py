import logging
import time
from typing import Optional

import cv2
from picamera2 import Picamera2
from tflite_support.task import core, processor, vision

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path: str = 'efficientdet_lite0.tflite', num_threads: int = 4, display_width: int = 1280, display_height: int = 72):
        try:
            self.pi_cam_2 = Picamera2()
            self.pi_cam_2.preview_configuration.main.size = (display_width, display_height)
            self.pi_cam_2.preview_configuration.main.format = 'RGB888'
            self.pi_cam_2.preview_configuration.align()
            self.pi_cam_2.configure("preview")
            self.pi_cam_2.start()

            self.web_cam = '/dev/video2'
            self.cam = cv2.VideoCapture(self.web_cam)

            if not self.cam.isOpened():
                raise RuntimeError(f"Could not open camera {self.web_cam}")

            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, display_width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, display_height)
            self.cam.set(cv2.CAP_PROP_FPS, 30)

            base_options = core.BaseOptions(file_name=model_path, use_coral=False, num_threads=num_threads)
            detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.2)
            options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
            self.detector = vision.ObjectDetector.create_from_options(options)

            self.font = cv2.FONT_HERSHEY_SIMPLEX
            self.pos = (20, 60)
            self.height = 1.5
            self.weight = 0.5
            self.my_color = (0, 255, 0)
            self.box_color = (255, 0, 0)

            self.fps = 0
            self.time_start = time.time()

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    def _process_frame(self, frame: cv2.Mat) -> Optional[cv2.Mat]:
        """Process single frame for object detection"""
        try:
            if frame is None:
                logger.warning("Received empty frame")
                return None

            frame_flipped = cv2.flip(frame, -1)
            image_rgb = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2RGB)
            image_tensor = vision.TensorImage.create_from_array(image_rgb)

            detections = self.detector.detect(image_tensor)

            for detect in detections.detections:
                bbox = detect.bounding_box
                upper_left = (bbox.origin_x, bbox.origin_y)
                lower_right = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
                obj_name = detect.categories[0].category_name

                cv2.rectangle(frame, upper_left, lower_right, self.box_color, 2)
                cv2.putText(frame, obj_name, upper_left, self.font, self.weight, self.my_color)

            # Calculate and display FPS
            time_elapsed = time.time() - self.time_start
            self.fps = 0.9 * self.fps + 0.1 * 1 / (time_elapsed or 0.001)
            cv2.putText(frame, f"{int(self.fps)} FPS", (20, 40), self.font, 1, self.my_color, 2)

            self.time_start = time.time()
            return frame

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None

    def run(self):
        """Main detection loop"""
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue

                processed_frame = self._process_frame(frame)

                if processed_frame is not None:
                    cv2.imshow('Object Detection', processed_frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            logger.info("Detection stopped by user")
        except Exception as e:
            logger.error(f"Unexpected error in detection loop: {e}")
        finally:
            self.cam.release()
            cv2.destroyAllWindows()


def main():
    try:
        detector = ObjectDetector()
        detector.run()
    except Exception as e:
        logger.error(f"Failed to start object detection: {e}")


if __name__ == "__main__":
    main()
