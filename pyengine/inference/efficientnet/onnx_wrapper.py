from typing import List, Tuple, Dict, Optional

import cv2  # For image resizing
import numpy as np
import onnxruntime


class OnnxWrapper:
    """
    Wraps an ONNX model (e.g., EfficientNet) for inference, providing an API
    similar to the TensorRT C API wrapper.
    """
    def __init__(self, model_path: str, max_batch_size: int = 1,
                 providers: List[str] = ['CPUExecutionProvider']):
        """
        Initializes the ONNX Runtime session and prepares for batching.

        Args:
            model_path: Path to the ONNX model file.
            max_batch_size: The maximum number of images that can be processed
                            in a single inference call.
            providers: List of ONNX Runtime execution providers to use
                       (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
                       Defaults to CPU.
        """
        if max_batch_size <= 0:
             raise ValueError("max_batch_size must be positive.")

        self.max_batch_size = max_batch_size
        self.session_options = onnxruntime.SessionOptions()
        # Optional: Enable optimizations
        # self.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            print(f"Loading ONNX model from: {model_path}")
            print(f"Using Execution Providers: {providers}")
            self.session = onnxruntime.InferenceSession(
                model_path,
                sess_options=self.session_options,
                providers=providers
            )
            print("ONNX model loaded successfully.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

        # Get model input details
        model_inputs = self.session.get_inputs()
        if not model_inputs:
             raise RuntimeError("Could not get model inputs from ONNX file.")
        self.input_name = model_inputs[0].name
        input_shape = model_inputs[0].shape # e.g., [None, 3, 224, 224] or [1, 3, 224, 224]

        # Handle dynamic batch size in input shape (replace None/dynamic axis with max_batch_size for planning)
        # We need target H and W for preprocessing
        try:
            # Assuming shape is [batch, channels, height, width]
            self.input_height = input_shape[2]
            self.input_width = input_shape[3]
            if not isinstance(self.input_height, int) or not isinstance(self.input_width, int):
                raise ValueError("Model input height/width must be fixed dimensions.")
            print(f"Model expects input shape like: [Batch, Channels, {self.input_height}, {self.input_width}]")
        except (IndexError, TypeError, ValueError) as e:
             print(f"Warning: Could not reliably determine input H/W from shape {input_shape}. Assuming 224x224. Error: {e}")
             # Default for many EfficientNets if shape is weird
             self.input_height = 224
             self.input_width = 224


        # Get model output details
        model_outputs = self.session.get_outputs()
        if not model_outputs:
             raise RuntimeError("Could not get model outputs from ONNX file.")
        self.output_name = model_outputs[0].name # Assuming single output

        # Internal buffer for batching
        # Stores tuples of (original_index, preprocessed_image_tensor)
        self._batch_buffer: List[Tuple[int, np.ndarray]] = []
        # Store results after inference, mapping original_index to result array
        self._results: Dict[int, np.ndarray] = {}


    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single image (HWC, BGR, uint8) for EfficientNet.
        **NOTE:** Adjust this based on your specific model's requirements!
        """
        # 1. Resize
        img_resized = cv2.resize(image, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)

        # 2. BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 3. Normalize (Example: scale to [0, 1] and apply ImageNet normalization)
        # Scale to [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0

        # ImageNet mean and std dev (RGB order)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_normalized = (img_float - mean) / std

        # 4. HWC to CHW (Height, Width, Channels to Channels, Height, Width)
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # 5. Add batch dimension (CHW -> NCHW where N=1)
        img_batch = np.expand_dims(img_chw, axis=0) # Shape: [1, C, H, W]

        return img_batch.astype(np.float32) # Ensure correct dtype

    def release(self):
        """
        Releases resources. For ONNX Runtime, this primarily clears buffers.
        The session itself is managed by Python's garbage collector.
        """
        print("Releasing ONNX wrapper resources...")
        self.session = None # Allow session to be garbage collected if no other refs
        self._batch_buffer.clear()
        self._results.clear()


    def add_image(self, index: int, image: np.ndarray) -> bool:
        """
        Preprocesses and adds an image to the current batch buffer.

        Args:
            index: An integer identifier for this image (used in get_result).
            image: The input image (NumPy array, expected HWC, BGR, uint8).

        Returns:
            True if the image was added successfully, False if the batch is full.
        """
        if self.session is None:
            print("Error: Session has been released.")
            return False
        if len(self._batch_buffer) >= self.max_batch_size:
            print(f"Warning: Batch buffer is full (max size: {self.max_batch_size}). Cannot add image index {index}.")
            return False

        # Basic validation of input image shape
        if len(image.shape) != 3 or image.shape[2] != 3:
             print(f"Error: Image for index {index} has unexpected shape {image.shape}. Expected HWC BGR.")
             return False

        try:
            preprocessed_image = self._preprocess(image)
            # Check if preprocessed shape matches model expectation (excluding batch dim)
            expected_shape_chw = (preprocessed_image.shape[1], self.input_height, self.input_width)
            if preprocessed_image.shape[1:] != expected_shape_chw:
                 print(f"Warning: Preprocessed image shape {preprocessed_image.shape[1:]} doesn't match model expected CHW {expected_shape_chw}")
                 # Optionally raise an error or try to continue if it's just padding differences

            self._batch_buffer.append((index, preprocessed_image))
            return True
        except Exception as e:
            print(f"Error preprocessing image for index {index}: {e}")
            return False


    def inference(self) -> bool:
        """
        Performs inference on the current batch of images.
        Clears the batch buffer afterwards. Stores results internally.

        Returns:
            True if inference was successful, False otherwise.
        """
        if self.session is None:
            print("Error: Session has been released.")
            return False
        if not self._batch_buffer:
            print("Warning: Inference called with empty batch buffer.")
            return False # Or maybe True, as technically nothing failed? Let's say False.

        # Prepare batch tensor
        # Concatenate the individual preprocessed images (each shaped [1, C, H, W])
        # into a single batch tensor [N, C, H, W]
        batch_input_tensor = np.concatenate([item[1] for item in self._batch_buffer], axis=0)
        actual_batch_size = batch_input_tensor.shape[0]

        # Store the original indices in the order they appear in the batch
        original_indices = [item[0] for item in self._batch_buffer]

        # Clear results from previous inference run
        self._results.clear()

        try:
            # Run inference
            input_feed = {self.input_name: batch_input_tensor}
            # session.run returns a list of output arrays (usually just one for classification)
            outputs = self.session.run([self.output_name], input_feed)

            if outputs and len(outputs) > 0:
                output_batch = outputs[0] # Get the first output tensor [N, NumClasses]
                # Ensure the output batch size matches the input batch size
                if output_batch.shape[0] == actual_batch_size:
                     # Store results mapped by original index
                     for i in range(actual_batch_size):
                         self._results[original_indices[i]] = output_batch[i] # Store result for this index
                     print(f"Inference successful for batch of {actual_batch_size}.")
                     return True
                else:
                    print(f"Error: Output batch size ({output_batch.shape[0]}) != Input batch size ({actual_batch_size})")
                    return False
            else:
                 print("Error: Inference did not return any outputs.")
                 return False

        except Exception as e:
            print(f"Error during ONNX inference: {e}")
            return False
        finally:
            # Always clear the buffer after attempting inference
            self._batch_buffer.clear()


    def get_result(self, item_index: int) -> Optional[np.ndarray]:
        """
        Retrieves the inference result for a specific image index from the last batch.

        Args:
            item_index: The original index of the image provided to add_image.

        Returns:
            A NumPy array containing the model's output for that image,
            or None if the index is not found or inference failed.
            The output shape depends on the model (e.g., [NumClasses] for classification).
        """
        if self.session is None:
            print("Error: Session has been released.")
            return None

        result_array = self._results.get(item_index) # Retrieve from dictionary

        if result_array is not None:
            # Return a copy to prevent external modification of internal results
            return np.copy(result_array)
        else:
            # print(f"Result for index {item_index} not found in last inference run.")
            return None


# Example Usage (Illustrative)
if __name__ == "__main__":
    # Replace with the actual path to your ONNX model
    onnx_model_path = "/path/to/your/efficientnet_model.onnx"
    max_b = 4
    # Use ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU if available and compatible
    # Note: Ensure onnxruntime-gpu is installed and CUDA/cuDNN are set up correctly.
    providers_to_use = ['CPUExecutionProvider']

    try:
        print("Initializing ONNX Engine Wrapper...")
        engine = OnnxEngineWrapper(onnx_model_path, max_batch_size=max_b, providers=providers_to_use)

        # --- Simulate adding images (replace with actual image loading) ---
        # Create dummy images matching expected input size (e.g., HWC BGR)
        dummy_h, dummy_w = 250, 300 # Example dimensions BEFORE preprocessing
        # Need input H/W from engine for dummy data generation matching preprocessing expectations
        in_h, in_w = engine.input_height, engine.input_width

        images_to_add = {}
        print(f"\nPreparing {max_b} dummy images...")
        for i in range(max_b):
             # Create a simple gradient image
             img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
             img[:, :, 0] = int((i / max_b) * 255) # Vary blue channel
             img[:, :, 1] = 128 # Constant green
             img[:, :, 2] = int(((max_b - 1 - i) / max_b) * 255) # Vary red channel
             images_to_add[i] = img # Map index i to this image

        print("Adding images to batch...")
        added_count = 0
        for idx, img_data in images_to_add.items():
            success = engine.add_image(idx, img_data)
            if success:
                print(f"  Added image with index {idx}")
                added_count += 1
            else:
                print(f"  Failed to add image with index {idx}")

        # --- Perform Inference ---
        if added_count > 0:
            print("\nPerforming inference...")
            infer_success = engine.inference()

            # --- Get Results ---
            if infer_success:
                print("\nRetrieving results:")
                for i in range(max_b): # Try to get results for original indices
                    result = engine.get_result(i)
                    if result is not None:
                        # EfficientNet typically outputs logits for classification
                        print(f"  Result for index {i}: shape={result.shape}, dtype={result.dtype}")
                        # Example: print top class prediction
                        # top_class_index = np.argmax(result)
                        # print(f"    Top prediction index: {top_class_index}, Logit value: {result[top_class_index]:.4f}")
                    else:
                        print(f"  Result for index {i}: Not found (was it added successfully?)")
            else:
                print("Inference failed.")
        else:
             print("\nNo images were added to the batch, skipping inference.")

        # --- Release ---
        print("\nReleasing engine...")
        engine.release()

    except FileNotFoundError:
        print(f"Error: Model file not found at {onnx_model_path}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")