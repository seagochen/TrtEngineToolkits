import ctypes
import traceback
from typing import List

import numpy as np

from pyengine.inference.c_wrapper.c_pose_data_struct import C_Inference_Result, C_Extended_Person_Feats, \
    InferenceResult, PoseDetection, Rect, Point
from pyengine.utils.logger import logger


class PosePipelineWrapper:
    """
    Python wrapper around the C/C++ pose detection inference engine
    loaded via ctypes. Handles common library loading, function mapping,
    and data conversion.
    """

    def __init__(self,
                 pose_model_path: str = "/opt/models/yolov8s-pose.engine",
                 feats_model_path: str = "/opt/models/efficientnet_b0_feat_logits.engine",
                 c_lib_path: str = '/opt/TrtEngineToolkits/lib/libjetson.so',
                 maximum_det_items: int = 100,
                 cls_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        """
        Initializes the PosePipelineWrapper with the given model paths and parameters.

        :param pose_model_path: Path to the YOLO pose detection model engine.
        :param feats_model_path: Path to the EfficientNet feature extraction model engine.
        :param c_lib_path: Path to the C shared library (.so file).
        :param maximum_det_items: Maximum number of detection items to process per image.
                                  (Note: The C header comment mentions YoloV8=8, EfficientNet=32 as optimal batch sizes.
                                  This `maximum_det_items` seems to be for max detections *per image* within a batch,
                                  not the batch size itself for the C API. Clarify if needed.)
        :param cls_threshold: Classification confidence threshold for detections.
        :param iou_threshold: Intersection over Union threshold for non-maximum suppression.

        :raises RuntimeError: If the C shared library cannot be loaded or engine initialization fails.
        """
        self._is_initialized = False
        self._pose_detection_lib = None

        # Load the shared library
        try:
            self._pose_detection_lib = ctypes.cdll.LoadLibrary(c_lib_path)
            logger.info("PosePipelineWrapper", f"Loaded C shared library from: {c_lib_path}")
        except OSError as e:
            logger.error("PosePipelineWrapper", f"Error loading shared library {c_lib_path}: {e}")
            raise RuntimeError(f"Failed to load C shared library: {e}")

        # Define C API function signatures
        self._define_c_apis()

        # Initialize the C engine
        try:
            c_pose_engine_path = pose_model_path.encode("utf-8")
            c_feats_engine_path = feats_model_path.encode("utf-8")

            if not self._pose_detection_lib.init_pose_detection_pipeline(
                    c_pose_engine_path,
                    c_feats_engine_path,
                    maximum_det_items,
                    cls_threshold,
                    iou_threshold
            ):
                logger.error("PosePipelineWrapper", "C engine initialization failed.")
                raise RuntimeError("C engine initialization failed.")

            self._is_initialized = True
            logger.info("PosePipelineWrapper", "C engine initialization succeeded.")

        except Exception as e:
            logger.error("PosePipelineWrapper", f"Error during C engine initialization: {e}")
            self.release() # Attempt to release resources if initialization failed partially
            raise RuntimeError(f"C engine initialization failed: {e}")

    def _define_c_apis(self):
        """
        Defines the argtypes and restypes for the C API functions.
        """
        # bool init_pose_detection_pipeline(const char* yolo_engine_path, const char* efficient_engine_path,
        #                                  int max_items, float cls, float iou);
        self._pose_detection_lib.init_pose_detection_pipeline.argtypes = [
            ctypes.c_char_p,  # yolo_engine_path
            ctypes.c_char_p,  # efficient_engine_path
            ctypes.c_int,     # max_items
            ctypes.c_float,   # cls
            ctypes.c_float    # iou
        ]
        self._pose_detection_lib.init_pose_detection_pipeline.restype = ctypes.c_bool

        # void add_image_to_pose_detection_pipeline(const unsigned char* image_data_in_bgr, int width, int height);
        self._pose_detection_lib.add_image_to_pose_detection_pipeline.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte), # image_data_in_bgr
            ctypes.c_int,                   # width
            ctypes.c_int                    # height
        ]
        self._pose_detection_lib.add_image_to_pose_detection_pipeline.restype = None

        # bool run_pose_detection_pipeline(C_Inference_Result** out_results, int *out_num_results);
        self._pose_detection_lib.run_pose_detection_pipeline.argtypes = [
            ctypes.POINTER(ctypes.POINTER(C_Inference_Result)), # out_results (pointer to pointer)
            ctypes.POINTER(ctypes.c_int)                      # out_num_results (pointer to int)
        ]
        self._pose_detection_lib.run_pose_detection_pipeline.restype = ctypes.c_bool

        # void deinit_pose_detection_pipeline();
        self._pose_detection_lib.deinit_pose_detection_pipeline.argtypes = []
        self._pose_detection_lib.deinit_pose_detection_pipeline.restype = None

        # void release_inference_result(C_Inference_Result* result_array, int count);
        self._pose_detection_lib.release_inference_result.argtypes = [
            ctypes.POINTER(C_Inference_Result), # result_array (pointer to array start)
            ctypes.c_int                        # count (number of elements in array)
        ]
        self._pose_detection_lib.release_inference_result.restype = None

    def add_image(self, image: np.ndarray) -> None:
        """
        Adds an image to the C engine's internal queue for processing.
        This function does not immediately perform detection.

        :param image: Image data as a numpy array. Expected shape (H, W, 3) and dtype np.uint8 (BGR).
                      Must be C-contiguous.
        :raises ValueError: If the image is not 3-channel (BGR) or not C-contiguous.
        :raises RuntimeError: If the pipeline is not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call __init__() first.")

        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Image must be a 3-channel (BGR) numpy array of np.uint8. "
                             f"Got shape {image.shape}, dtype {image.dtype}")

        if not image.flags['C_CONTIGUOUS']:
            logger.warning("PosePipelineWrapper", "Image is not C-contiguous. Attempting to make a copy.")
            image = np.ascontiguousarray(image)
            # If conversion to contiguous fails or is too slow,
            # you might consider raising an error here instead of a warning.


        height, width, _ = image.shape
        # Get a pointer to the raw byte data of the numpy array
        image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

        self._pose_detection_lib.add_image_to_pose_detection_pipeline(image_ptr, width, height)

    def inference(self) -> List[InferenceResult]:
        """
        Performs pose detection and information enrichment on all queued images.
        This is a blocking call. The internal queue will be cleared after execution.

        :return: A list of `InferenceResult` objects, where each object represents
                 the detection results for one image.
        :raises RuntimeError: If the pipeline is not initialized or detection fails in the C layer.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call __init__() first.")

        # Allocate pointers to receive results from C
        out_results_ptr = ctypes.POINTER(C_Inference_Result)()
        out_num_results = ctypes.c_int(0)

        success = False
        try:
            success = self._pose_detection_lib.run_pose_detection_pipeline(
                ctypes.byref(out_results_ptr),
                ctypes.byref(out_num_results)
            )

            if not success:
                logger.error("PosePipelineWrapper", "Failed to run pose detection pipeline in C layer.")
                raise RuntimeError("Failed to run pose detection pipeline.")

            num_images_processed = out_num_results.value
            python_results: List[InferenceResult] = []

            if num_images_processed > 0 and out_results_ptr:
                for i in range(num_images_processed):
                    c_inference_result: C_Inference_Result = out_results_ptr[i]
                    image_detections: List[PoseDetection] = []

                    if c_inference_result.num_detected > 0 and c_inference_result.detections:
                        for j in range(c_inference_result.num_detected):
                            c_detection: C_Extended_Person_Feats = c_inference_result.detections[j]

                            # Convert C_Rect to Rect
                            py_rect = Rect(
                                x1=float(c_detection.box.x1),
                                y1=float(c_detection.box.y1),
                                x2=float(c_detection.box.x2),
                                y2=float(c_detection.box.y2),
                            )

                            # Convert C_Point array to List[Point]
                            py_keypoints = [
                                Point(x=float(p.x), y=float(p.y), score=float(p.score))
                                for p in c_detection.pts
                            ]

                            # Convert features C array to List[float]
                            py_features = [float(f) for f in c_detection.features]

                            # Create PoseDetection object
                            py_pose_detection = PoseDetection(
                                box=py_rect,
                                confidence=float(c_detection.confidence),
                                class_id=int(c_detection.class_id),
                                keypoints=py_keypoints,
                                features=py_features
                            )
                            image_detections.append(py_pose_detection)

                    # Create InferenceResult object for the current image
                    py_inference_result = InferenceResult(
                        num_detected=int(c_inference_result.num_detected),
                        detections=image_detections
                    )
                    python_results.append(py_inference_result)
            return python_results
        except Exception as e:
            logger.error("PosePipelineWrapper", f"Error during inference: {e}\n{traceback.format_exc()}")
            raise
        finally:
            if success and out_results_ptr:
                try:
                    self._pose_detection_lib.release_inference_result(out_results_ptr, out_num_results.value)
                    # logger.debug("PosePipelineWrapper", "C-allocated inference results released.")

                except Exception as release_e:
                    logger.error("PosePipelineWrapper", f"Error releasing C-allocated memory: {release_e}")


    def release(self):
        """
        Releases the underlying C engine resources.
        """
        if self._is_initialized and self._pose_detection_lib:
            try:
                self._pose_detection_lib.deinit_pose_detection_pipeline()
                self._is_initialized = False
                logger.info("PosePipelineWrapper", "Released C engine resources.")
            except Exception as e:
                logger.error("PosePipelineWrapper", f"Error during deinitialization of C engine: {e}")
        else:
            logger.info("PosePipelineWrapper", "Pipeline not initialized or already released.")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point. Ensures resources are released."""
        logger.info("PosePipelineWrapper", "Exiting PosePipelineWrapper context. Releasing resources.")
        self.release()

    def is_initialized(self):
        return self._is_initialized


"""
# --- Example Usage (for testing the class) ---
import cv2
import os

from pyengine.utils.logger import logger
from pyengine.inference.c_wrapper.c_pose_pipeline_wrapper import PosePipelineWrapper
from pyengine.inference.c_wrapper.c_pose_data_struct import Point, Rect, PoseDetection, InferenceResult


if __name__ == "__main__":
    # Define the paths of engines and libs
    YOLO_POSE_ENGINE = "/opt/models/yolov8s-pose.engine"
    EFFICIENTNET_ENGINE = "/opt/models/efficientnet_b0_feat_logits.engine"
    C_LIB = "/opt/TrtEngineToolkits/lib/libjetson.so"
    OUTPUT_DIR = "output/pose_crops"  # Directory to save cropped images

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n--- Visualizing Pose Detection Results ---")

    # Mock logger if not fully set up in your environment
    if not hasattr(logger, 'info'):
        class MockLogger:
            def info(self, tag, msg): print(f"[INFO][{tag}] {msg}")
            def error(self, tag, msg): print(f"[ERROR][{tag}] {msg}")
            def warning(self, tag, msg): print(f"[WARNING][{tag}] {msg}")
            def debug(self, tag, msg): print(f"[DEBUG][{tag}] {msg}")
        logger = MockLogger()

    pipeline = None
    try:
        # Use the context manager for automatic resource release
        with PosePipelineWrapper(
            pose_model_path=YOLO_POSE_ENGINE,
            feats_model_path=EFFICIENTNET_ENGINE,
            c_lib_path=C_LIB,
            maximum_det_items=100,
            cls_threshold=0.4,
            iou_threshold=0.3
        ) as pipeline:

            print("\nPipeline initialized.")

            # Load the images for testing
            test_image_paths = [
                "/opt/images/supermarket/customer1.png",
                "/opt/images/supermarket/customer2.png",
                "/opt/images/supermarket/customer3.png",
                "/opt/images/supermarket/customer4.png",
                "/opt/images/supermarket/customer5.png",
                "/opt/images/supermarket/customer6.png",
                "/opt/images/supermarket/customer7.png",
                "/opt/images/supermarket/customer8.png",
                "/opt/images/supermarket/staff1.png",
                "/opt/images/supermarket/staff2.png",
                "/opt/images/supermarket/staff3.png",
                "/opt/images/supermarket/staff4.png",
                "/opt/images/supermarket/staff5.png",
                "/opt/images/supermarket/staff6.png",
                "/opt/images/supermarket/staff7.png",
                "/opt/images/supermarket/staff8.png",
            ]

            loaded_images = {}
            for img_path in test_image_paths:
                if not os.path.exists(img_path):
                    logger.warning("Example", f"Image file not found: {img_path}. Skipping.")
                    continue
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (640, 640))
                    if img is None:
                        logger.error("Example", f"Failed to read image from {img_path}. Skipping.")
                        continue
                    loaded_images.update({img_path: img})
                    print(f"Adding image {img_path} of shape {img.shape} to pipeline...")
                    pipeline.add_image(img)
                except Exception as e:
                    logger.error("Example", f"Error processing image {img_path}: {e}")

            print(f"\nAdded {len(loaded_images)} successfully loaded images to the pipeline.")

            # Run inference on the added images
            print("\nRunning inference...")
            results = pipeline.inference()

            print(f"\nInference completed. Received results for {len(results)} images.")

            for i, (img_path, original_image) in enumerate(loaded_images.items()):
                image_results = results.pop(0) if results else None
                if image_results:
                    image_filename = os.path.basename(img_path).split('.')[0]
                    for person_index, detection in enumerate(image_results.detections):
                        box = detection.box
                        class_id = int(detection.class_id)  # Ensure class_id is an integer
                        keypoints = detection.keypoints

                        # Crop the person
                        x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
                        cropped_person = original_image.copy()[y1:y2, x1:x2]

                        if cropped_person.size > 0:
                            # Draw red keypoints on the cropped person
                            for kp in keypoints:
                                cx, cy = int(kp.x - x1), int(kp.y - y1)
                                cv2.circle(cropped_person, (cx, cy), 5, (0, 0, 255), -1)  # Red circle

                            # Create the filename
                            output_filename = f"{image_filename}_person_{person_index + 1}_class_{class_id}.png"
                            output_path = os.path.join(OUTPUT_DIR, output_filename)

                            # Save the cropped image
                            cv2.imwrite(output_path, cropped_person)
                            print(f"Saved cropped person with keypoints to: {output_path}")
                        else:
                            logger.warning("Example", f"Cropped person has zero size for {image_filename}, person {person_index + 1}.")
                else:
                    logger.warning("Example", f"No results found for image: {img_path}")

    except RuntimeError as e:
        logger.error("Example", f"Caught a runtime error: {e}")
    except Exception as e:
        logger.error("Example", f"An unexpected error occurred: {e}")
    finally:
        print("\n--- Visualization finished ---")
        if pipeline and pipeline.is_initialized():
            print("Note: Pipeline was still initialized after example, explicitly calling release.")
            pipeline.release()
"""
