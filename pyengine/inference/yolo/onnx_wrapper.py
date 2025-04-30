from typing import Union, List, Tuple, Dict, Optional

import numpy as np
import cv2
import onnxruntime

from pyengine.inference.yolo.d_struct.data_struct import YoloPose, Yolo, YoloPoint


class YoloOnnxWrapper:
    """
    Wraps a YOLO/YOLO-Pose ONNX model for inference, providing an API
    similar to the C API wrapper. Handles preprocessing and postprocessing.
    """
    def __init__(self, model_path: str, use_pose: bool = False,
                 providers: List[str] = ['CPUExecutionProvider']):
        """
        Initializes the ONNX Runtime session for YOLO.

        Args:
            model_path: Path to the ONNX model file.
            use_pose: Set True if the model is a YOLO-Pose model.
            providers: List of ONNX Runtime execution providers.
        """
        self.use_pose = use_pose
        self.session_options = onnxruntime.SessionOptions()

        try:
            print(f"Loading YOLO ONNX model from: {model_path}")
            print(f"Using Execution Providers: {providers}")
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=providers
            )
            print("YOLO ONNX model loaded successfully.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

        # Get model input details
        model_inputs = self.session.get_inputs()
        if not model_inputs:
            raise RuntimeError("Could not get model inputs from ONNX file.")
        self.input_name = model_inputs[0].name
        input_shape = model_inputs[0].shape # e.g., [1, 3, 640, 640]

        try:
            # Assuming shape is [batch, channels, height, width]
            self.input_height = int(input_shape[2])
            self.input_width = int(input_shape[3])
            print(f"Model expects input shape: [Batch, Channels, {self.input_height}, {self.input_width}]")
        except Exception as e:
            raise RuntimeError(f"Could not determine input H/W from shape {input_shape}: {e}")

        # Get model output details
        model_outputs = self.session.get_outputs()
        if not model_outputs:
            raise RuntimeError("Could not get model outputs from ONNX file.")
        # Assuming single output, common in YOLOv5/v8 exports
        self.output_name = model_outputs[0].name
        # Example output shape: [batch, 4+1+num_classes (+kpts*3), num_detections] for YOLOv8
        self.output_shape = model_outputs[0].shape
        print(f"Model output name: {self.output_name}, shape: {self.output_shape}")
        # Determine number of classes and keypoints from output shape if possible
        # This depends heavily on the exact YOLO version output format!
        # Example for YOLOv8 detection: shape = [b, 84, 8400] -> 84 = 4 (box) + 80 (classes) -> num_classes=80
        # Example for YOLOv8 pose: shape = [b, 56, 8400] -> 56 = 4 (box) + 1 (class) + 51 (17*3 kpts) -> num_classes=1
        try:
            output_dims = self.output_shape[1] # e.g., 84 or 56
            if self.use_pose:
                # Assuming 1 class, 17 keypoints for pose -> 4 + 1 + 17*3 = 56
                if output_dims == 56:
                     self.num_classes = 1
                     self.num_keypoints = 17
                     print("Interpreting output as YOLOv8-Pose (1 class, 17 kpts)")
                else:
                     # Heuristic: Try to guess num_keypoints
                     # Need at least 4 bbox + 1 conf + 1 class = 6 dims
                     if output_dims > 6 and (output_dims - 5) % 3 == 0:
                         self.num_keypoints = (output_dims - 5) // 3
                         self.num_classes = 1 # Assume 1 class if kpts present? Risky guess.
                         print(f"Warning: Assuming YOLO-Pose with {self.num_keypoints} kpts and 1 class based on output dim {output_dims}")
                     else:
                         raise ValueError(f"Cannot determine pose structure from output dim {output_dims}")

            else: # Detection
                 self.num_keypoints = 0
                 # Need at least 4 bbox + 1 conf = 5 dims
                 if output_dims > 5:
                     self.num_classes = output_dims - 4 # YOLOv8 structure assumption
                     print(f"Interpreting output as YOLOv8-Detection ({self.num_classes} classes)")
                 else:
                     raise ValueError(f"Cannot determine detection structure from output dim {output_dims}")

        except Exception as e:
            # Set defaults, user might need to override
            self.num_classes = 80 if not use_pose else 1
            self.num_keypoints = 17 if use_pose else 0


        # Internal buffer for batching
        # Stores (original_index, preprocessed_image_tensor, original_hw_tuple)
        self._batch_buffer: List[Tuple[int, np.ndarray, Tuple[int, int]]] = []
        # Stores raw output tensor from the last inference
        self._raw_results: Optional[np.ndarray] = None
        # Stores metadata associated with the raw results (indices, shapes)
        self._inference_meta: Dict = {'indices': [], 'original_hw': []}
        # Stores post-processed results, keyed by original image index
        # Value is the list of Yolo/YoloPose objects for that image
        self._result_map: Dict[int, List[Union[Yolo, YoloPose]]] = {}
        # Stores the index for which available_results was last called
        self._last_available_index: Optional[int] = None


    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, int, int]]:
        """
        Preprocesses a single image (HWC, BGR, uint8) for YOLOv8.
        Includes letterboxing and scaling.

        Returns:
            Tuple: (preprocessed_image_tensor [1, C, H, W], (scale_ratio_h, scale_ratio_w, pad_h, pad_w))
        """
        img_h, img_w = image.shape[:2]
        target_h, target_w = self.input_height, self.input_width

        # Calculate scaling ratio and new size
        ratio = min(target_w / img_w, target_h / img_h)
        new_unpad_w, new_unpad_h = int(round(img_w * ratio)), int(round(img_h * ratio))
        pad_w, pad_h = (target_w - new_unpad_w) / 2, (target_h - new_unpad_h) / 2

        # Resize and pad
        if (new_unpad_w, new_unpad_h) != (img_w, img_h):
            img_resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = image

        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # Pad with gray

        # BGR to RGB, HWC to CHW, Normalize to [0, 1]
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB) # Convert color
        img_chw = np.transpose(img_rgb, (2, 0, 1)).astype(np.float32) # HWC -> CHW
        img_normalized = img_chw / 255.0 # Scale to [0,1]

        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0) # Shape: [1, C, H, W]

        # Store necessary info for postprocessing scaling
        scale_ratio_h = ratio
        scale_ratio_w = ratio
        pad_top = top
        pad_left = left

        return img_batch.astype(np.float32), (scale_ratio_h, scale_ratio_w, pad_top, pad_left)

    def _postprocess(self,
                    raw_output_batch: np.ndarray,
                    inference_meta: Dict,
                    conf_threshold: float,
                    nms_threshold: float) -> Dict[int, List[Union[Yolo, YoloPose]]]:
        """
        Postprocesses the raw output batch from YOLOv8 ONNX model.
        Includes confidence filtering, NMS, and coordinate scaling.

        Args:
            raw_output_batch: The raw output tensor from session.run(), e.g., shape [N, 56, 8400] or [N, 84, 8400].
            inference_meta: Dictionary containing 'indices' and 'original_hw' lists for the batch.
            conf_threshold: Confidence threshold for filtering detections.
            nms_threshold: IoU threshold for Non-Maximum Suppression.

        Returns:
            Dictionary mapping original image index to a list of detected Yolo/YoloPose objects.
        """
        processed_results = {}
        num_images_in_batch = raw_output_batch.shape[0]
        original_indices = inference_meta['indices']
        original_hw_list = inference_meta['original_hw']
        scaling_pads_list = inference_meta['scaling_pads'] # Get scaling info

        # Output format expected: [batch, 4_xywh + 1_conf + num_classes (+ kpts*3), num_proposals]
        # Transpose to [batch, num_proposals, 4+1+classes (+kpts*3)] for easier processing
        outputs = np.transpose(raw_output_batch, (0, 2, 1))

        for i in range(num_images_in_batch):
            output_single = outputs[i] # Shape [num_proposals, dims]
            original_h, original_w = original_hw_list[i]
            scale_h, scale_w, pad_top, pad_left = scaling_pads_list[i] # Get scaling/padding for this image

            # Filter by confidence
            # For detection: box_conf * class_conf
            # For pose: box_conf (class assumed 0?)
            if self.use_pose:
                # Pose: output[:, 4] is object confidence
                conf_scores = output_single[:, 4]
                valid_indices = np.where(conf_scores >= conf_threshold)[0]
            else:
                # Detection: Find max class score and its index
                class_scores = output_single[:, 4:] # Scores for all classes
                max_scores = np.max(class_scores, axis=1)
                box_conf = output_single[:, 4] # Objectness score often not present in YOLOv8 output, use max_class
                overall_conf = max_scores # Use max class score as confidence
                valid_indices = np.where(overall_conf >= conf_threshold)[0]

            if len(valid_indices) == 0:
                processed_results[original_indices[i]] = []
                continue

            # Select valid proposals
            filtered_output = output_single[valid_indices, :]
            valid_conf_scores = overall_conf[valid_indices] if not self.use_pose else conf_scores[valid_indices]
            valid_class_ids = np.argmax(filtered_output[:, 4:], axis=1) if not self.use_pose else np.zeros(len(valid_indices), dtype=int) # Assume class 0 for pose

            # Extract boxes (cx, cy, w, h) - model coordinates (relative to input size like 640x640)
            boxes_model_xywh = filtered_output[:, :4]

            # Convert xywh to xyxy (lx, ly, rx, ry) - still model coordinates
            boxes_model_xyxy = np.zeros_like(boxes_model_xywh)
            boxes_model_xyxy[:, 0] = boxes_model_xywh[:, 0] - boxes_model_xywh[:, 2] / 2 # lx
            boxes_model_xyxy[:, 1] = boxes_model_xywh[:, 1] - boxes_model_xywh[:, 3] / 2 # ly
            boxes_model_xyxy[:, 2] = boxes_model_xywh[:, 0] + boxes_model_xywh[:, 2] / 2 # rx
            boxes_model_xyxy[:, 3] = boxes_model_xywh[:, 1] + boxes_model_xywh[:, 3] / 2 # ry

            # Extract keypoints if pose model - model coordinates
            keypoints_model = None
            if self.use_pose and filtered_output.shape[1] >= 5 + self.num_keypoints * 3:
                 # kpts data starts after box (4) and conf (1, assuming only 1 class)
                 kpts_raw = filtered_output[:, 5:].reshape(len(valid_indices), self.num_keypoints, 3) # [N, K, 3] x, y, conf(or visibility)
                 keypoints_model = kpts_raw


            # --- Apply NMS ---
            # Use cv2.dnn.NMSBoxes. Requires boxes as (x, y, w, h) and scores.
            # We use xyxy format internally, so let's adapt or use a different NMS if needed.
            # NMSBoxes works with xywh format list of lists/tuples.
            boxes_for_nms = [ [int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes_model_xywh ]
            indices_after_nms = cv2.dnn.NMSBoxes(boxes_for_nms, valid_conf_scores.tolist(), conf_threshold, nms_threshold)

            # Filter results after NMS
            final_detections = []
            if len(indices_after_nms) > 0:
                 # Handle scalar vs list output of NMSBoxes
                 if isinstance(indices_after_nms, np.ndarray):
                    indices_after_nms = indices_after_nms.flatten()

                 for idx in indices_after_nms:
                     box_xyxy = boxes_model_xyxy[idx]
                     conf = valid_conf_scores[idx]
                     cls_id = valid_class_ids[idx]
                     kpts_single = keypoints_model[idx] if keypoints_model is not None else None

                     # --- Scale coordinates back to original image size ---
                     # 1. Remove padding
                     lx = box_xyxy[0] - pad_left
                     ly = box_xyxy[1] - pad_top
                     rx = box_xyxy[2] - pad_left
                     ry = box_xyxy[3] - pad_top
                     # 2. Scale back to original size
                     lx /= scale_w
                     ly /= scale_h
                     rx /= scale_w
                     ry /= scale_h
                     # 3. Clip to image bounds
                     lx_orig = max(0, int(lx))
                     ly_orig = max(0, int(ly))
                     rx_orig = min(original_w, int(rx))
                     ry_orig = min(original_h, int(ry))

                     # Check if box is valid after scaling
                     if rx_orig <= lx_orig or ry_orig <= ly_orig:
                         continue

                     if self.use_pose and kpts_single is not None:
                         pts_orig = []
                         for kpt_idx in range(self.num_keypoints):
                             kpt_x_model, kpt_y_model, kpt_conf = kpts_single[kpt_idx]
                             # Scale keypoints similar to boxes
                             kpt_x = (kpt_x_model - pad_left) / scale_w
                             kpt_y = (kpt_y_model - pad_top) / scale_h
                             # Clip keypoints (optional, but good practice)
                             kpt_x_orig = max(0, min(original_w, int(kpt_x)))
                             kpt_y_orig = max(0, min(original_h, int(kpt_y)))
                             # Store scaled keypoint
                             pts_orig.append(YoloPoint(x=kpt_x_orig, y=kpt_y_orig, conf=float(kpt_conf)))

                         final_detections.append(YoloPose(lx=lx_orig, ly=ly_orig, rx=rx_orig, ry=ry_orig,
                                                          cls=int(cls_id), conf=float(conf), pts=pts_orig))
                     else: # Detection
                         final_detections.append(Yolo(lx=lx_orig, ly=ly_orig, rx=rx_orig, ry=ry_orig,
                                                     cls=int(cls_id), conf=float(conf)))

            processed_results[original_indices[i]] = final_detections

        return processed_results


    def release(self):
        """Releases ONNX Runtime session and clears buffers."""
        print("Releasing YOLO ONNX wrapper resources...")
        self.session = None
        self._batch_buffer.clear()
        self._raw_results = None
        self._result_map.clear()
        self._inference_meta.clear()
        self._last_available_index = None

    def add_image(self, index: int, image: np.ndarray) -> bool:
        """Adds an image to the batch buffer after preprocessing."""
        if self.session is None:
            print("Error: Session has been released.")
            return False
        if len(self._batch_buffer) >= self.max_batch_size:
            print(f"Warning: Batch buffer full (max: {self.max_batch_size}). Cannot add index {index}.")
            return False
        if len(image.shape) != 3 or image.shape[2] != 3:
            print(f"Error: Image index {index} has shape {image.shape}. Expected HWC BGR.")
            return False

        original_hw = image.shape[:2] # Store H, W
        try:
            preprocessed_tensor, scale_pad_info = self._preprocess(image)
            self._batch_buffer.append((index, preprocessed_tensor, original_hw, scale_pad_info))
            return True
        except Exception as e:
            print(f"Error preprocessing image index {index}: {e}")
            return False

    def inference(self) -> bool:
        """Performs inference on the current batch."""
        if self.session is None:
            print("Error: Session has been released.")
            return False
        if not self._batch_buffer:
            print("Warning: Inference called with empty batch buffer.")
            return False

        # Prepare batch input
        batch_input_tensor = np.concatenate([item[1] for item in self._batch_buffer], axis=0)
        actual_batch_size = batch_input_tensor.shape[0]

        # Store metadata needed for postprocessing
        self._inference_meta['indices'] = [item[0] for item in self._batch_buffer]
        self._inference_meta['original_hw'] = [item[2] for item in self._batch_buffer]
        self._inference_meta['scaling_pads'] = [item[3] for item in self._batch_buffer] # Store scaling info

        # Clear previous results before running
        self._raw_results = None
        self._result_map.clear()
        self._last_available_index = None

        try:
            input_feed = {self.input_name: batch_input_tensor}
            outputs = self.session.run([self.output_name], input_feed)

            if outputs and len(outputs) > 0:
                self._raw_results = outputs[0] # Store raw output [N, Dims, Proposals]
                print(f"Inference successful for batch of {actual_batch_size}.")
                return True
            else:
                print("Error: Inference did not return outputs.")
                return False
        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            return False
        finally:
            # Clear input buffer AFTER inference attempt
            self._batch_buffer.clear()

    def available_results(self, index: int, cls_thresh: float, nms_thresh: float) -> int:
        """
        Triggers postprocessing on the results of the last inference run
        using the provided thresholds and returns the number of detections
        found for the specified image index.

        Args:
            index: The original index of the image to check results for.
            cls_thresh: Confidence threshold for filtering.
            nms_thresh: IoU threshold for NMS.

        Returns:
            The number of detected objects (Yolo/YoloPose) for the given index
            after postprocessing, or 0 if none found or error occurred.
        """
        self._last_available_index = index # Remember which index was queried

        if self.session is None or self._raw_results is None:
            print("Error: Cannot get available results. Session released or inference not run.")
            return 0

        # Perform postprocessing *every time* this is called, as thresholds might change
        try:
            self._result_map = self._postprocess(self._raw_results, self._inference_meta, cls_thresh, nms_thresh)
        except Exception as e:
            print(f"Error during postprocessing for available_results: {e}")
            self._result_map = {} # Clear results if postprocessing fails
            return 0

        # Return the count for the requested index
        count = len(self._result_map.get(index, []))
        # print(f"Available results for index {index} (conf>{cls_thresh}, nms>{nms_thresh}): {count}")
        return count

    def get_result(self, item_index: int) -> Optional[Union[Yolo, YoloPose]]:
        """
        Retrieves a specific detection result by its index within the list
        of results obtained for the image index last queried by available_results().

        Args:
            item_index: The 0-based index of the detection within the results list
                        for the last queried image.

        Returns:
            The Yolo or YoloPose object for the specified detection, or None if
            the index is out of bounds or results are unavailable.
        """
        if self.session is None:
            print("Error: Session has been released.")
            return None
        if self._last_available_index is None:
            print("Error: available_results() must be called before get_result().")
            return None

        # Get the list of results for the last queried image index
        results_list = self._result_map.get(self._last_available_index)

        if results_list is not None and 0 <= item_index < len(results_list):
            return results_list[item_index] # Return the specific detection object
        else:
            # print(f"Result item index {item_index} out of bounds for image index {self._last_available_index} (found {len(results_list) if results_list is not None else 0}).")
            return None


# Example Usage
if __name__ == "__main__":
    # --- Configuration ---
    # !!! Replace with the actual path to YOUR YOLO ONNX model !!!
    onnx_yolo_model_path = "/path/to/your/yolov8n.onnx" # Or yolov8n-pose.onnx
    is_pose_model = False # Set to True if using a YOLO-Pose model
    max_batch = 4
    conf_thr = 0.45
    nms_thr = 0.5
    # Use ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU if available
    providers_to_use = ['CPUExecutionProvider']

    try:
        print("Initializing YOLO ONNX Wrapper...")
        yolo_engine = YoloOnnxWrapper(onnx_yolo_model_path, use_pose=is_pose_model, providers=providers_to_use)

        # --- Simulate adding images ---
        # Create dummy images matching expected input size
        in_h, in_w = yolo_engine.input_height, yolo_engine.input_width
        images_to_add = {}
        print(f"\nPreparing {max_batch} dummy images (input size: {in_h}x{in_w})...")
        for i in range(max_batch):
             # Create a simple gradient image (HWC BGR)
             img = np.zeros((in_h + 20, in_w - 10, 3), dtype=np.uint8) # Use different original size
             img[:, :, 0] = int((i / max_batch) * 255)
             img[:, :, 1] = 128
             img[:, :, 2] = int(((max_batch - 1 - i) / max_batch) * 255)
             images_to_add[i] = img

        print("Adding images...")
        added_count = 0
        for idx, img_data in images_to_add.items():
            if yolo_engine.add_image(idx, img_data):
                print(f"  Added image index {idx} (Original shape: {img_data.shape[:2]})")
                added_count += 1
            else:
                print(f"  Failed to add image index {idx}")

        # --- Perform Inference ---
        if added_count > 0:
            print("\nPerforming inference...")
            infer_success = yolo_engine.inference()

            # --- Get Results (Iterate through original images) ---
            if infer_success:
                print(f"\nRetrieving results (Conf > {conf_thr}, NMS IoU < {nms_thr})...")
                for i in range(max_batch): # Check results for each image added
                    num_results = yolo_engine.available_results(i, conf_thr, nms_thr)
                    print(f"  Image index {i}: Found {num_results} results.")
                    for res_idx in range(num_results):
                        result_obj = yolo_engine.get_result(res_idx)
                        if result_obj:
                            print(f"    Result {res_idx}: Type={type(result_obj).__name__}, Class={result_obj.cls}, Conf={result_obj.conf:.3f}, Box=[{result_obj.lx},{result_obj.ly},{result_obj.rx},{result_obj.ry}]")
                            if isinstance(result_obj, YoloPose):
                                print(f"      Num Keypoints: {len(result_obj.pts)}")
                                # print(f"      First Keypoint: ({result_obj.pts[0].x}, {result_obj.pts[0].y}), Conf={result_obj.pts[0].conf:.2f}")
            else:
                print("Inference failed.")
        else:
             print("\nNo images added, skipping inference.")

        # --- Release ---
        print("\nReleasing engine...")
        yolo_engine.release()

    except FileNotFoundError:
        print(f"Error: Model file not found at {onnx_yolo_model_path}")
        print("Please update the 'onnx_yolo_model_path' variable.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()