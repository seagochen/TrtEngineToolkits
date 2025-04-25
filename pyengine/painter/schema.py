from pyengine.utils.load_schema import KeyPoint, Skeleton


class SchemaLoader:

    def __init__(self, schema_file: str = None):
        self.bbox_colors = self.default_schema()
        if schema_file:
            self.load_external_schema(schema_file)
        else:
            self.default_schema()

    def default_schema(self):
        bbox_colors = [
            (255, 0, 0),  # Class 0: Blue
            (0, 255, 0),  # Class 1: Green
            (0, 0, 255),  # Class 2: Red
            (255, 255, 0),  # Class 3: Cyan
            (255, 0, 255),  # Class 4: Magenta
            (0, 255, 255),  # Class 5: Yellow
            (128, 0, 128),  # Class 6: Purple
            (128, 128, 0),  # Class 7: Olive
            (128, 128, 128),  # Class 8: Gray
            (0, 128, 255)  # Class 9: Orange
        ]

        return bbox_colors

    # def load_external_schema(self, schema_file: str):
    #     """加载外部的关键点和骨骼映射。"""
    #     if not os.path.isfile(schema_file):
    #         raise FileNotFoundError("The schema file does not exist.")

    #     kpt_color_map, skeleton_map, bbox_colors = load_schema_from_json(schema_file)
    #     self.kpt_color_map = kpt_color_map
    #     self.skeleton_map = skeleton_map
    #     self.bbox_colors = bbox_colors