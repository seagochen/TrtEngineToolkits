from typing import List, Tuple
from pyengine.inference.unified_structs.inference_results import Rect, Skeleton, Point


def scale_euler_pt(
        src_width: int, src_height:
        int, dst_width: int, dst_height: int,
        point: Tuple[int, int]) -> Tuple[int, int]:
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    return int(point[0] * scale_x), int(point[1] * scale_y)


def scale_euler_pts(
    src_width: int, src_height: int,
    dst_width: int, dst_height: int,
    points: List[Tuple[float, float]]
) -> List[Tuple[int, int]]:
    """
    将一组点从原始尺寸缩放到目标尺寸
    
    :param src_width: 原始宽度
    :param src_height: 原始高度
    :param dst_width: 目标宽度
    :param dst_height: 目标高度
    :param points: [(x1, y1), (x2, y2), ...]
    :return: 缩放后的点列表
    """
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    
    return [
        (int(x * scale_x), int(y * scale_y))
        for x, y in points
    ]


def scale_sk_pt(src_width: int, src_height: int, dst_width: int, dst_height: int, point: Point) -> Point:
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    return Point(int(point.x * scale_x), int(point.y * scale_y), point.confidence)


def scale_sk_rect(src_width: int, src_height: int, dst_width: int, dst_height: int, rect: Rect) -> Rect:
    scale_x = dst_width / src_width
    scale_y = dst_height / src_height
    return Rect(
        int(rect.x1 * scale_x),
        int(rect.y1 * scale_y),
        int(rect.x2 * scale_x),
        int(rect.y2 * scale_y)
    )


def scale_sk_pts(src_width: int, src_height: int, dst_width: int, dst_height: int, points: list) -> list:
    """将归一化的点列表缩放到像素坐标"""
    out = []
    for point in points:
        out.append(scale_sk_pt(src_width, src_height, dst_width, dst_height, point))
    return out


def scale_skeleton(src_width: int, src_height: int, dst_width: int, dst_height: int,
                   skeleton: Skeleton, inplace: bool = False) -> Skeleton:
    """将归一化的骨骼 Skeleton 缩放到像素坐标"""

    # 缩放关键点
    scaled_points = scale_sk_pts(src_width, src_height, dst_width, dst_height, skeleton.points)
    scaled_rect = scale_sk_rect(src_width, src_height, dst_width, dst_height, skeleton.rect)

    if inplace:
        skeleton.points = scaled_points
        skeleton.rect = scaled_rect
    else:
        skeleton = Skeleton(
            rect=scaled_rect,
            points=scaled_points,
            track_id=skeleton.track_id,
            classification=skeleton.classification,
            confidence=skeleton.confidence,
            features=skeleton.features.copy() if skeleton.features else []
        )
    return skeleton


def scale_skeletons(src_width: int, src_height: int, dst_width: int, dst_height: int,
                    skeletons: list, inplace: bool = False) -> list:
    """将归一化的骨骼列表缩放到像素坐标"""

    scaled_skeletons = []
    for skeleton in skeletons:
        scaled_skeletons.append(scale_skeleton(src_width, src_height, dst_width, dst_height, skeleton, inplace))
    return scaled_skeletons