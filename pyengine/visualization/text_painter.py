import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os # Added for font path check

def calculate_average_color(image, bbox):
    """计算边界框内的平均颜色 (BGR)。"""
    # 确保 bbox 在图像范围内
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)

    # 检查裁剪区域是否有效
    if x2 <= x1 or y2 <= y1:
        # 返回一个默认值，例如图像中心像素或纯灰色
        center_y, center_x = img_h // 2, img_w // 2
        # print(f"Warning: Invalid bbox {bbox} for image shape {(img_h, img_w)}. Using center pixel.")
        # 检查图像是否为灰度图
        if len(image.shape) == 2 or image.shape[2] == 1:
             return np.array([image[center_y, center_x]] * 3) # 复制灰度值到3个通道
        else:
             return image[center_y, center_x]
        # return np.array([128, 128, 128]) # 或者返回灰色

    cropped_image = image[y1:y2, x1:x2]
    # 处理可能的空裁剪区域（如果bbox完全在图像外）
    if cropped_image.size == 0:
        # print(f"Warning: Cropped image is empty for bbox {bbox}. Using default color.")
        return np.array([128, 128, 128]) # 返回灰色

    # 计算平均颜色
    # 检查图像是否为灰度图
    if len(cropped_image.shape) == 2 or cropped_image.shape[2] == 1:
         avg_color_gray = np.mean(cropped_image)
         return np.array([avg_color_gray] * 3) # 复制灰度值到3个通道
    else:
         # 对于彩色图像
         # OpenCV 使用 BGR 顺序
         average_color = np.average(np.average(cropped_image, axis=0), axis=0)
         return average_color


def decide_text_color(average_color):
    """根据平均背景颜色决定使用黑色还是白色文本。"""
    # 计算亮度 (常用加权平均)
    # 对于 BGR: Luminance = 0.114*B + 0.587*G + 0.299*R
    luminance = 0.114 * average_color[0] + 0.587 * average_color[1] + 0.299 * average_color[2]
    # print(f"Average BGR: {average_color}, Luminance: {luminance}")
    return (0, 0, 0) if luminance > 127 else (255, 255, 255) # BGR format


def draw_text(frame: np.ndarray,
              text: str,
              left_top: tuple, # (x, y) for the top-left corner of the text area
              font_path: str = None, # Path to .ttf or .otf font file for Pillow/UTF-8
              font_size: int = 20,   # Font size for Pillow
              font_scale: float = 0.7, # Font scale for OpenCV
              thickness: int = 1,    # Thickness for OpenCV
              text_color: tuple = None, # BGR color tuple (B, G, R). None for auto-contrast.
              background_color: tuple = None, # BGR color tuple for background. None for no background.
              background_alpha: float = 0.5, # Transparency for background (0=transparent, 1=opaque)
              background_padding: int = 5     # Padding around text for background
             ) -> np.ndarray:
    """
    在图像上绘制文本，支持 ASCII (OpenCV) 和 UTF-8 (Pillow)。
    自动选择绘制方法基于 font_path 是否提供。
    支持自动文本颜色对比和带透明背景的文本。

    Args:
        frame: OpenCV 图像 (NumPy array BGR).
        text: 要绘制的文本字符串.
        left_top: 文本区域左上角的 (x, y) 坐标.
        font_path: (可选) .ttf 或 .otf 字体文件的路径。如果提供，则使用 Pillow 进行绘制 (支持 UTF-8)。
        font_size: (可选) 当使用 Pillow 时使用的字体大小。
        font_scale: (可选) 当使用 OpenCV 时使用的字体缩放比例。
        thickness: (可选) 当使用 OpenCV 时使用的线条粗细。
        text_color: (可选) BGR 格式的文本颜色 (B, G, R)。如果为 None，则自动选择黑色或白色以获得对比度。
        background_color: (可选) BGR 格式的背景颜色。如果为 None，则不绘制背景。
        background_alpha: (可选) 背景颜色的透明度 (0.0 到 1.0)。
        background_padding: (可选) 文本周围背景的填充大小（像素）。

    Returns:
        带有绘制文本的 OpenCV 图像 (NumPy array BGR)。
    """
    use_pillow = font_path is not None
    final_text_color = text_color # Store user specified or later decided color

    if use_pillow:
        # --- 使用 Pillow (UTF-8 / 自定义字体) ---
        try:
            # 检查字体文件是否存在
            if not os.path.exists(font_path):
                 raise FileNotFoundError(f"Font file not found at: {font_path}")

            # 加载字体
            font = ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"Error loading font: {e}. Falling back to OpenCV default.")
            use_pillow = False # 回退到 OpenCV
            # 尝试使用 OpenCV 的默认字体绘制错误消息或原始文本
            cv2.putText(frame, f"Font Error: {text[:20]}...", left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            return frame

        # 计算文本边界框 (Pillow)
        # 创建一个临时的 ImageDraw 对象来获取文本大小
        try:
            # 需要一个临时的Image来创建Draw对象
            temp_img = Image.new('RGB', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            # textbbox 返回 (left, top, right, bottom) 相对于给定坐标
            text_bbox = temp_draw.textbbox((0, 0), text, font=font) # Calculate size based on origin (0,0)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            # Pillow的textbbox可能包含一些空白，有时textlength更精确宽度
            # text_width = temp_draw.textlength(text, font=font)
        except Exception as e:
             print(f"Error calculating text size with Pillow: {e}")
             # 使用一个估计值或回退
             text_width = len(text) * font_size // 2 # 非常粗略的估计
             text_height = font_size
             # 或者完全回退到OpenCV显示错误
             cv2.putText(frame, f"Size Error: {text[:20]}...", left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
             return frame


        # --- 自动颜色决策 (如果需要) ---
        if final_text_color is None:
            # 定义用于颜色计算的区域 (文本将出现的地方)
            color_bbox = (left_top[0], left_top[1], text_width, text_height)
            avg_bg_color = calculate_average_color(frame, color_bbox)
            final_text_color = decide_text_color(avg_bg_color)
            # print(f"Pillow - Auto text color decided: {final_text_color}")

        # 将 OpenCV 图像转换为 PIL 格式 (BGR -> RGB)
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil, 'RGBA' if background_color is not None and background_alpha < 1.0 else 'RGB')

        # --- 绘制背景 (如果需要) ---
        if background_color is not None:
            bg_left = left_top[0] - background_padding
            bg_top = left_top[1] - background_padding
            # 注意：Pillow 的 bbox 可能不从 0 开始，所以我们使用计算出的宽高
            bg_right = left_top[0] + text_width + background_padding
            bg_bottom = left_top[1] + text_height + background_padding
            background_rect = (bg_left, bg_top, bg_right, bg_bottom)

            # 创建一个单独的层来绘制带 alpha 的矩形
            overlay = Image.new('RGBA', frame_pil.size, (255, 255, 255, 0)) # 透明层
            draw_overlay = ImageDraw.Draw(overlay)

            # BGR -> RGB
            bg_color_rgb = (background_color[2], background_color[1], background_color[0])
            bg_alpha_int = int(background_alpha * 255)
            draw_overlay.rectangle(background_rect, fill=bg_color_rgb + (bg_alpha_int,))

            # 将覆盖层混合到主图像上
            frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), overlay).convert('RGB')
            # 更新 draw 对象，因为它现在基于新的 frame_pil
            draw = ImageDraw.Draw(frame_pil)


        # --- 绘制文本 (Pillow) ---
        text_color_rgb = (final_text_color[2], final_text_color[1], final_text_color[0]) # BGR -> RGB
        draw.text(left_top, text, font=font, fill=text_color_rgb)

        # 将 PIL 图像转换回 OpenCV 格式 (RGB -> BGR)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    else:
        # --- 使用 OpenCV (ASCII / 默认字体) ---
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # 计算文本大小 (OpenCV)
        (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
        # baseline 是文本基线相对于文本框底部的高度

        # --- 自动颜色决策 (如果需要) ---
        if final_text_color is None:
            # 定义用于颜色计算的区域
            # 注意：left_top 是左上角，但 OpenCV 的 putText 使用左下角
            # 我们需要覆盖从 left_top[1] 到 left_top[1] + text_height 的区域
            color_bbox = (left_top[0], left_top[1], text_width, text_height + baseline) # 稍微扩大以包含基线区域
            avg_bg_color = calculate_average_color(frame, color_bbox)
            final_text_color = decide_text_color(avg_bg_color)
            # print(f"OpenCV - Auto text color decided: {final_text_color}")


        # --- 绘制背景 (如果需要) ---
        if background_color is not None:
            # 定义背景矩形区域 (左上角和右下角)
            bg_top_left = (left_top[0] - background_padding,
                           left_top[1] - background_padding)
            bg_bottom_right = (left_top[0] + text_width + background_padding,
                               left_top[1] + text_height + baseline + background_padding) # 包含基线

            if background_alpha < 1.0:
                # 使用 addWeighted 实现透明度
                overlay = frame.copy()
                cv2.rectangle(overlay, bg_top_left, bg_bottom_right, background_color, -1) # -1 表示填充
                cv2.addWeighted(overlay, background_alpha, frame, 1 - background_alpha, 0, frame)
            else:
                # 直接绘制不透明矩形
                cv2.rectangle(frame, bg_top_left, bg_bottom_right, background_color, -1)

        # --- 绘制文本 (OpenCV) ---
        # 计算 putText 的 org 参数 (左下角坐标)
        # 我们希望文本的顶部在 left_top[1]，所以基线应该在 left_top[1] + text_height
        text_origin = (left_top[0], left_top[1] + text_height)
        cv2.putText(frame, text, text_origin, font_face, font_scale, final_text_color, thickness, lineType=cv2.LINE_AA)

    return frame

# --- Example Usage ---
if __name__ == "__main__":

    # 加载图像
    # image_path = "/opt/images/spectrum2.png" # 使用你的图像路径
    image_path = "spectrum.png" # 假设图像在当前目录
    if not os.path.exists(image_path):
         print(f"Error: Image file not found at {image_path}. Creating a dummy image.")
         # 创建一个彩色渐变图像作为备用
         height, width = 400, 600
         frame = np.zeros((height, width, 3), dtype=np.uint8)
         for i in range(height):
             for j in range(width):
                 frame[i, j, 0] = int(i / height * 255) # Blue channel gradient
                 frame[i, j, 1] = int(j / width * 255) # Green channel gradient
                 frame[i, j, 2] = int((i+j)/(height+width)*255) # Red channel gradient
    else:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Unable to load image file: {image_path}")
            exit()

    # --- 字体设置 ---
    # 尝试找到一个系统字体或指定你自己的字体路径
    # 对于 Linux/macOS:
    font_paths_to_try = [
        "/usr/share/fonts/truetype/ Noto_Sans_JP/NotoSansJP-Regular.ttf", # 常见 Linux 路径
        "/System/Library/Fonts/Arial Unicode.ttf", # macOS
        "C:/Windows/Fonts/arial.ttf", # Windows
        "fonts/NotoSansJP-Regular.ttf", # 本地目录 (如果存在)
        "arial.ttf" # 如果在系统路径中
    ]
    utf8_font_path = None
    for p in font_paths_to_try:
        if os.path.exists(p):
            utf8_font_path = p
            print(f"Using font: {utf8_font_path}")
            break

    if utf8_font_path is None:
        print("Warning: No suitable TTF font found for UTF-8 rendering. Falling back to OpenCV for all text.")


    # --- 绘制示例 ---

    # 1. 使用 OpenCV (ASCII) - 自动颜色
    frame = draw_text(frame, "ASCII Text (OpenCV Auto Color)", (30, 50),
                      font_scale=0.8, thickness=2)

    # 2. 使用 OpenCV (ASCII) - 指定颜色 + 背景
    frame = draw_text(frame, "ASCII with Background (OpenCV)", (30, 100),
                      font_scale=0.8, thickness=2, text_color=(255, 255, 0), # Cyan text
                      background_color=(100, 0, 0), background_alpha=0.7) # Semi-transparent dark blue BG

    # 3. 使用 Pillow (UTF-8) - 自动颜色 (如果找到字体)
    if utf8_font_path:
        frame = draw_text(frame, "UTF-8 中文 Japanese 日本語 (Pillow Auto Color)", (30, 150),
                          font_path=utf8_font_path, font_size=25)
    else:
         frame = draw_text(frame, "UTF-8 requires TTF font (Not Found)", (30, 150), font_scale=0.6, text_color=(0, 0, 255))


    # 4. 使用 Pillow (UTF-8) - 指定颜色 + 背景 (如果找到字体)
    if utf8_font_path:
        frame = draw_text(frame, "UTF-8 with Background (Pillow)", (30, 200),
                          font_path=utf8_font_path, font_size=25,
                          text_color=(0, 255, 0), # Green text
                          background_color=(50, 50, 50), # Dark Gray BG
                          background_alpha=0.8, background_padding=8)
    else:
         frame = draw_text(frame, "Pillow BG needs TTF font (Not Found)", (30, 200), font_scale=0.6, text_color=(0, 0, 255))

    # 5. 自动颜色在复杂背景上的测试
    frame = draw_text(frame, "Auto Color Test", (250, 280),
                       font_path=utf8_font_path, font_size=40) # Use Pillow if available

    # 调整显示大小
    display_height = 600
    h, w = frame.shape[:2]
    scale = display_height / h
    display_width = int(w * scale)
    frame_display = cv2.resize(frame, (display_width, display_height))

    # 显示结果
    cv2.imshow('Combined Text Drawing Demo', frame_display)
    print("Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()