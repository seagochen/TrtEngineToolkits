from dataclasses import dataclass
import cv2


@dataclass
class InferenceCache:
    index: int
    image: any = None
    inference: any = None


class MqttCache:

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}  # Stores InferenceCache objects indexed by frame numbers

    def add_image(self, image, index):
        """Add an image to the cache by index."""
        if index not in self.cache:
            self.cache[index] = InferenceCache(index=index)
        self.cache[index].image = image

    def add_inference(self, inference, index):
        """Add inference result to the cache by index."""
        if index not in self.cache:
            self.cache[index] = InferenceCache(index=index)
        self.cache[index].inference = inference

    def get_min_index(self):
        """Find and return the InferenceCache entry with the smallest index."""
        if not self.cache:
            return None  # Cache is empty

        # Find the minimum index key
        min_index = min(self.cache.keys())
        return min_index

    def has_image(self, index):
        """Check if the cache has an image for the given index."""
        return index in self.cache and self.cache[index].image is not None
    
    def has_inference(self, index):
        """Check if the cache has an inference result for the given index."""
        return index in self.cache and self.cache[index].inference is not None
    
    def complete(self, index):
        """Check if the cache has both an image and an inference result for the given index."""
        return self.has_image(index) and self.has_inference(index)
    
    def count(self):
        """Return the number of entries in the cache."""
        return len(self.cache)
    
    def get(self, index):
        """Return the InferenceCache entry for the given index."""
        # Get the index entry from the cache if it exists
        return self.cache.get(index)
    
    def remove(self, index):
        """Remove the InferenceCache entry for the given index."""
        if index in self.cache:
            del self.cache[index]
