import ctypes
import numpy as np
import cv2  # Import cv2 for image resizing and type checks


# 假设你的 logger 模块在这里
from pyengine.utils.logger import logger
# 如果没有实际的 logger 模块，可以定义一个简单的占位符
# class SimpleLogger:
#     def info(self, tag, message):
#         print(f"[INFO] <{cv2.getTickCount() / cv2.getTickFrequency():.6f}> [{tag}] - {message}")
#
#     def debug(self, tag, message):
#         print(f"[DEBUG] <{cv2.getTickCount() / cv2.getTickFrequency():.6f}> [{tag}] - {message}")
#
#     def error(self, tag, message):
#         print(f"[ERROR] <{cv2.getTickCount() / cv2.getTickFrequency():.6f}> [{tag}] - {message}")
#
#
# logger = SimpleLogger()


# --- 1. 定义C结构体映射 ---

class C_KeyPoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("conf", ctypes.c_float),
    ]


class C_YoloPose(ctypes.Structure):
    _fields_ = [
        ("lx", ctypes.c_int),
        ("ly", ctypes.c_int),
        ("rx", ctypes.c_int),
        ("ry", ctypes.c_int),
        ("cls", ctypes.c_int),
        ("num_pts", ctypes.c_int),
        ("conf", ctypes.c_float),
        ("pts", ctypes.POINTER(C_KeyPoint)),  # 指向C_KeyPoint数组的指针
        ("feats", ctypes.POINTER(ctypes.c_float)),  # 指向float数组（长度256）的指针
    ]


class C_ImagePoseResults(ctypes.Structure):
    _fields_ = [
        ("image_idx", ctypes.c_int),
        ("num_detections", ctypes.c_int),
        ("detections", ctypes.POINTER(C_YoloPose)),  # 指向C_YoloPose数组的指针
    ]


class C_BatchedPoseResults(ctypes.Structure):
    _fields_ = [
        ("num_images", ctypes.c_int),
        ("results", ctypes.POINTER(C_ImagePoseResults)),  # 指向C_ImagePoseResults数组的指针
    ]


# --- 2. 加载共享库和定义函数签名 ---

class PosePipeline:
    def __init__(self, library_path):
        """
        初始化PosePipeline，加载C语言的共享库。
        """

        try:
            # 加载共享库
            self.lib = ctypes.CDLL(library_path)
            logger.info("PosePipeline", f"Shared library loaded successfully from {library_path}.")
        except OSError as e:
            raise RuntimeError(f"无法加载共享库 {library_path}。错误信息: {e}")

        # 定义函数签名
        self._define_function_signatures()

        # C端pipeline的上下文句柄
        self._context = None
        self._context_handle_ptr = ctypes.POINTER(ctypes.c_void_p)()  # 用于 create_pipeline 和 destroy_pipeline 时的双重指针

        # 其他参数
        self.max_yolo_batch_size = 0  # 最大批处理大小
        self.max_efficient_batch_size = 0  # 最大批处理大小

        # 用于存储处理后的图像数据，防止被Python垃圾回收
        self._processed_images_buffer = None

    def _define_function_signatures(self):
        """
        设置C函数的参数和返回值类型。
        """
        # void c_register_models();
        self.lib.c_register_models.argtypes = []
        self.lib.c_register_models.restype = None

        # YoloEfficientContext* c_create_pose_pipeline(...)
        self.lib.c_create_pose_pipeline.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_float, ctypes.c_float
        ]
        self.lib.c_create_pose_pipeline.restype = ctypes.c_void_p  # 不透明指针类型

        # C_BatchedPoseResults c_process_batched_images(...)
        self.lib.c_process_batched_images.argtypes = [
            ctypes.c_void_p,  # YoloEfficientContext* context, 直接传递 void*
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),  # const unsigned char* const* input_images_data
            ctypes.POINTER(ctypes.c_int),  # const int* widths
            ctypes.POINTER(ctypes.c_int),  # const int* heights
            ctypes.POINTER(ctypes.c_int),  # const int* channels
            ctypes.c_int  # int num_images
        ]
        self.lib.c_process_batched_images.restype = C_BatchedPoseResults

        # void c_free_batched_pose_results(C_BatchedPoseResults* results)
        self.lib.c_free_batched_pose_results.argtypes = [ctypes.POINTER(C_BatchedPoseResults)]
        self.lib.c_free_batched_pose_results.restype = None

        # void c_destroy_pose_pipeline(YoloEfficientContext* context)
        self.lib.c_destroy_pose_pipeline.argtypes = [ctypes.c_void_p]  # 传递 void*
        self.lib.c_destroy_pose_pipeline.restype = None

    def register_models(self):
        """
        注册姿态检测pipeline所需的模型。必须在创建上下文前调用一次。
        """
        logger.info("PosePipeline", "Registering models...")
        self.lib.c_register_models()
        logger.info("PosePipeline", "Models registered successfully.")

    def create_pipeline(self,
                        yolo_engine_path: str,  # YOLO模型的TensorRT引擎文件路径
                        efficient_engine_path: str,  # EfficientNet模型的TensorRT引擎文件路径
                        yolo_max_batch: int,  # YOLO模型的最大批处理大小
                        efficient_max_batch: int,  # EfficientNet模型的最大批处理大小
                        yolo_cls_thresh: float,  # YOLO模型的分类阈值
                        yolo_iou_thresh: float  # YOLO模型的IOU阈值
                        ):
        """
        初始化YoloPose和EfficientNet模型。

        Args:
            yolo_engine_path: YOLO模型的TensorRT引擎文件路径。
            efficient_engine_path: EfficientNet模型的TensorRT引擎文件路径。
            yolo_max_batch: YOLO模型的最大批处理大小。
            efficient_max_batch: EfficientNet模型的最大批处理大小。
            yolo_cls_thresh: YOLO模型的分类阈值。
            yolo_iou_thresh: YOLO模型的IOU阈值。
        """
        logger.info("PosePipeline", "Creating pipeline...")
        yolo_engine_path_c = yolo_engine_path.encode('utf-8')
        efficient_engine_path_c = efficient_engine_path.encode('utf-8')

        # C函数返回void* (YoloEfficientContext*)，ctypes会映射为Python的int
        # 这里需要将 _context_handle_ptr.contents 置为空，确保C函数返回新的地址
        self._context_handle_ptr.contents = ctypes.c_void_p(0)  # Initialize with null pointer

        context_address = self.lib.c_create_pose_pipeline(
            yolo_engine_path_c,
            efficient_engine_path_c,
            yolo_max_batch,
            efficient_max_batch,
            yolo_cls_thresh,
            yolo_iou_thresh
        )
        if context_address is None or context_address == 0:
            raise RuntimeError("创建姿态pipeline上下文失败。")

        # 记录最大批处理大小
        self.max_yolo_batch_size = yolo_max_batch
        self.max_efficient_batch_size = efficient_max_batch

        # 将context地址存储到 _context 中
        self._context = ctypes.cast(context_address, ctypes.c_void_p)

        # 同时，更新 _context_handle_ptr 的内容，以便 destroy_pipeline 可以使用它
        self._context_handle_ptr.contents = self._context

        logger.info("PosePipeline", "Pipeline created successfully.")

    def process_batched_images(self, images: list[np.ndarray]) -> list[dict]:
        """
        批量处理图像，经过YoloPose和EfficientNet pipeline。

        Args:
            images: NumPy数组列表，每个数组为一张图片，OpenCV格式：(H, W, C)，dtype=np.uint8 (BGR)。
                    注意：此函数将负责内部将图片统一resize到640x640。
        Returns:
            list[dict]: 返回一个包含每张图片检测结果的字典列表。
                            每个字典包含以下键：
                                - "image_idx": 图片索引
                                - "detections": 检测到的人体信息列表，每个元素是一个字典，包含：
                                    - "bbox": 人体边界框 [lx, ly, rx, ry]
                                    - "classification": 分类分数
                                    - "keypoints": 关键点列表，每个关键点是一个字典 {"x": x坐标, "y": y坐标, "conf": 置信度}
                                    - "features": 特征向量 (numpy数组)
        """

        if self._context is None:
            raise RuntimeError("姿态pipeline上下文未初始化。请先调用create_pipeline。")

        num_images = len(images)
        if num_images == 0:
            return []

        if num_images > self.max_yolo_batch_size:
            logger.error("PosePipeline",
                         f"批处理大小 ({num_images}) 超过YOLO模型的最大限制 {self.max_yolo_batch_size}。将处理前 {self.max_yolo_batch_size} 张图片。")
            images = images[:self.max_yolo_batch_size]  # Truncate if too many images
            num_images = len(images)

        # --------------------------- 为C函数准备输入数据 ---------------------------
        # C++ 端期望 RGB, HWC, 8-bit unsigned char.

        image_data_pointers = (ctypes.POINTER(ctypes.c_ubyte) * num_images)()
        widths = (ctypes.c_int * num_images)()
        heights = (ctypes.c_int * num_images)()
        channels = (ctypes.c_int * num_images)()

        # 存储调整大小后的图片，以便它们的内存不会被Python垃圾回收
        # 从而保证C++可以安全地访问这些数据
        self._processed_images_buffer = []

        for i, img in enumerate(images):
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 3:
                logger.error("PosePipeline", f"第{i}张图片必须是3维uint8类型的NumPy数组 (H, W, C)。")
                # 针对无效图片，填充占位符，或者跳过但调整num_images
                # 为简化，这里直接raise Error
                raise ValueError(f"第{i}张图片类型或维度不正确。")

            # 将图片调整大小到 640x640
            height, width = img.shape[:2]
            if height != 640 or width != 640:
                resized_img = cv2.resize(img, (640, 480))
            else:
                resized_img = img

            # 保证图片是C连续的，便于直接内存访问
            if not resized_img.flags['C_CONTIGUOUS']:
                resized_img = np.ascontiguousarray(resized_img)

            # 将处理后的图像存储在实例变量中，防止被GC
            self._processed_images_buffer.append(resized_img)

            # 填充 ctypes 数组
            image_data_pointers[i] = resized_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            heights[i], widths[i], channels[i] = resized_img.shape

        # 调用C函数
        # 传递 self._context (void*) 而不是 self._context_ptr (void**)
        c_batched_results = self.lib.c_process_batched_images(
            self._context,
            image_data_pointers,
            widths,
            heights,
            channels,
            num_images
        )

        # --------------------------- 处理C端结果 ---------------------------

        # 处理C端结果，转为Python对象
        python_results = []
        for i in range(c_batched_results.num_images):  # 这里的 c_batched_results 是 c_process_batched_images 的返回值

            # 获取每张图片的检测结果
            c_image_results = c_batched_results.results[i]

            # 解析检测结果
            image_detections = []
            for j in range(c_image_results.num_detections):

                # 首先是YoloPose的检测结果
                c_yolo_pose = c_image_results.detections[j]

                # 将C端的关键点转换为Python字典列表
                keypoints = []
                for k in range(c_yolo_pose.num_pts):
                    c_keypoint = c_yolo_pose.pts[k]
                    keypoints.append({
                        "x": c_keypoint.x,
                        "y": c_keypoint.y,
                        "conf": c_keypoint.conf
                    })

                # 从C端拷贝特征向量到numpy数组，长度为256
                # 检查feats指针是否有效
                feature_vector = np.array([])
                if c_yolo_pose.feats:  # 检查指针是否非空
                    feature_vector = np.array(
                        c_yolo_pose.feats[0:256], dtype=np.float32
                    )
                else:
                    logger.warning("PosePipeline",
                                   f"Detection {j} in image {i} has NULL feature pointer. Feature vector will be empty.")

                # 将检测结果转换为Python字典
                image_detections.append({
                    "bbox": [c_yolo_pose.lx, c_yolo_pose.ly, c_yolo_pose.rx, c_yolo_pose.ry],
                    "classification": c_yolo_pose.cls,
                    "confidence": c_yolo_pose.conf,
                    "keypoints": keypoints,
                    "features": feature_vector
                })

            # 将每张图片的结果添加到Python结果列表中
            python_results.append({
                "image_idx": c_image_results.image_idx,
                "detections": image_detections
            })

        # 释放C端分配的结果内存
        # **重要：使用 ctypes.byref() 传递 C_BatchedPoseResults 结构体本身的地址**
        self.lib.c_free_batched_pose_results(ctypes.byref(c_batched_results))
        # 清空内部图像缓冲区，以便GC可以回收内存
        self._processed_images_buffer = []

        return python_results

    def destroy_pipeline(self):
        """
        销毁YoloEfficientContext，释放所有相关模型资源。
        """
        if self._context is not None:
            logger.info("PosePipeline", "Destroying pose pipeline...")
            self.lib.c_destroy_pose_pipeline(self._context)
            self._context = None  # Python端也置为None
            self._context_handle_ptr.contents = ctypes.c_void_p(0)  # 也清空这个指针
            logger.info("PosePipeline", "Pose pipeline destroyed.")

    def __del__(self):
        """
        确保对象被回收时销毁pipeline。
        """
        self.destroy_pipeline()

