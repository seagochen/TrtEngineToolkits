import json

# 定义关键点结构
class KeyPoint:
    def __init__(self, name, color):
        self.name = name
        self.color = tuple(color)

    def __repr__(self):
        return f"KeyPoint(name={self.name}, color={self.color})"

# 定义骨骼连接结构
class Skeleton:
    def __init__(self, srt_kpt_id, dst_kpt_id, color):
        self.srt_kpt_id = srt_kpt_id
        self.dst_kpt_id = dst_kpt_id
        self.color = tuple(color)

    def __repr__(self):
        return f"Skeleton(srt_kpt_id={self.srt_kpt_id}, dst_kpt_id={self.dst_kpt_id}, color={self.color})"


# 加载并解析 JSON 文件的函数
def load_schema_from_json(filepath):
    """加载 JSON 文件并解析为 kpt_color_map, skeleton_map 和 bbox_colors."""
    with open(filepath, 'r') as file:
        data = json.load(file)

    # 解析 keypoint color map
    kpt_color_map = {int(k): KeyPoint(v["name"], v["color"]) for k, v in data["kpt_color_map"].items()}

    # 解析 skeleton map
    skeleton_map = [Skeleton(item["srt_kpt_id"], item["dst_kpt_id"], item["color"]) for item in data["skeleton_map"]]

    # 解析 bbox colors
    bbox_colors = [tuple(color_info["color"]) for color_info in data["bbox_color"]]

    return kpt_color_map, skeleton_map, bbox_colors


# 打印出加载后的数据
def print_loaded_data(kpt_color_map, skeleton_map, bbox_colors):
    print("Key Point Color Map:")
    for kpt_id, keypoint in kpt_color_map.items():
        print(f"  ID: {kpt_id}, {keypoint}")

    print("\nSkeleton Map:")
    for skeleton in skeleton_map:
        print(f"  {skeleton}")

    print("\nBounding Box Colors:")
    for idx, color in enumerate(bbox_colors):
        print(f"  Class {idx}: Color {color}")


# 示例用法
if __name__ == "__main__":
    # 假设 schema.json 是你的文件路径
    kpt_color_map, skeleton_map, bbox_colors = load_schema_from_json("../../configs/schema.json")

    # 打印加载的数据
    print_loaded_data(kpt_color_map, skeleton_map, bbox_colors)
