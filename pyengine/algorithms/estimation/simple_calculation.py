from pyengine.algorithms.estimation import is_valid_point, compute_modulus, analyze_front_side_back_face, \
    analyze_single_eye_face, analyze_back_face_ears_only
from pyengine.inference.unified_structs.auxiliary_structs import FaceDirection, ExtendedSkeleton
from pyengine.inference.unified_structs.inference_results import Skeleton

def calculate_direction_and_posture(skeleton: Skeleton) -> ExtendedSkeleton:
    """
    根据 pose_extend 中的面部关键点信息分析面部朝向，返回 FacialDirection 对象。
    """
    modulus = compute_modulus(skeleton, divisor=3.0)
    nose, right_eye, left_eye, right_ear, left_ear = skeleton.points[:5]

    valid_nose = is_valid_point(nose)
    valid_right_eye = is_valid_point(right_eye)
    valid_left_eye = is_valid_point(left_eye)
    valid_right_ear = is_valid_point(right_ear)
    valid_left_ear = is_valid_point(left_ear)

    orientation = FaceDirection.Unknown
    # (修改) 初始化角度变量
    angle = 0.0
    vec_x, vec_y = 0.0, 0.0
    origin_x, origin_y = 0.0, 0.0

    if valid_nose and valid_left_eye and valid_right_eye:
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_front_side_back_face(nose,
                                         left_eye,
                                         right_eye,
                                         left_ear,
                                         right_ear,
                                         valid_left_ear,
                                         valid_right_ear)

    elif valid_nose and (valid_left_eye or valid_right_eye):
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_single_eye_face(nose, left_eye, right_eye)

    elif valid_left_ear and valid_right_ear and not (valid_nose or valid_left_eye or valid_right_eye):
        # (修改) 接收返回的角度值
        orientation, angle, (vec_x, vec_y), (origin_x, origin_y) = \
            analyze_back_face_ears_only(left_ear, right_ear)

    extended_skeleton = ExtendedSkeleton(
        # 复制 Skeleton 的所有字段
        classification=skeleton.classification,
        confidence=skeleton.confidence,
        track_id=skeleton.track_id,
        features=skeleton.features,
        rect=skeleton.rect,
        points=skeleton.points,
        # 姿态检测, 不更新任何姿态相关字段
        # pose=Pose.Unknown,
        # 面部朝向
        direction=orientation,
        direction_angle=angle,
        direction_modulus=modulus,
        direction_vector=(vec_x, vec_y),
        direction_origin=(origin_x, origin_y)
    )
    return extended_skeleton