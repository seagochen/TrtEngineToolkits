import cv2
import time
from common.utils.logger import Logger

# Initialize the logger
logger = Logger()


class StreamReader:

    def __init__(self, url, width, height, fps, max_retries=5, delay=2):
        self.url = url
        self.cap = self.open_camera_stream(url)
        self.max_retries = max_retries
        self.delay = delay
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.last_frame_time = time.time()

        if self.cap:
            logger.info("StreamReader", f"Camera stream initialized successfully from {url}.")
        else:
            logger.error("StreamReader", f"Failed to initialize camera stream from {url}.")

    @staticmethod
    def open_camera_stream(url):
        """Open a camera stream and check if it's successful."""
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            logger.info("StreamReader", f"Camera stream opened successfully from {url}.")
            return cap
        else:
            logger.error("StreamReader", f"Failed to open camera stream from {url}.")
            return None

    def is_connected(self):
        connected = self.cap.isOpened()
        if connected:
            logger.debug("StreamReader", "Camera stream is connected.")
        else:
            logger.warning("StreamReader", "Camera stream is not connected.")
        return connected

    def read_frame(self):
        """Read a frame from the camera stream."""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_time:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (self.width, self.height))
                self.last_frame_time = current_time
                # logger.debug("StreamReader", "Frame read and resized successfully.")
                return frame
            else:
                # If the frame read fails, try to reconnect to the camera stream
                logger.warning("StreamReader", "Failed to read frame, attempting to reconnect.")
                self.cap = self.reconnect_camera(self.url, self.max_retries, self.delay)

                # If the reconnection is successful, read a frame
                if self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (self.width, self.height))
                        self.last_frame_time = current_time
                        logger.info("StreamReader", "Frame read successfully after reconnection.")
                        return frame

                # If the reconnection fails, return None
                logger.error("StreamReader", "Failed to read frame after reconnection.")
                return None
        else:
            # If the time elapsed since the last frame is less than the frame time, return None
            # logger.debug("StreamReader", "Frame skipped due to insufficient time since last frame.")
            return None

    def reconnect_camera(self, url, max_retries=5, delay=2):
        """Attempt to reconnect to the camera stream with retries."""
        attempts = 0
        while attempts < max_retries:
            cap = self.open_camera_stream(url)
            if cap:
                logger.info("StreamReader", f"Successfully reconnected to the camera on attempt {attempts + 1}.")
                return cap
            else:
                logger.warning("StreamReader", f"Reconnection attempt {attempts + 1}/{max_retries} failed. Retrying in {delay} seconds...")
                attempts += 1
                time.sleep(delay)

        logger.error("StreamReader", "Max retries reached. Could not reconnect to the camera.")
        return None

    def close_camera_stream(self):
        """Close the camera stream."""
        if self.cap:
            self.cap.release()
            logger.info("StreamReader", "Camera stream closed successfully.")
        else:
            logger.warning("StreamReader", "Attempted to close a camera stream that was not open.")

