import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def calculate_average_color(image, bbox):
    """Calculate the average color within a bounding box."""
    cropped_image = image[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    return np.average(np.average(cropped_image, axis=0), axis=0)

def decide_text_color(average_color):
    """Decide whether to use black or white text based on the average color."""
    return (0, 0, 0) if np.mean(average_color) > 127 else (255, 255, 255)

def draw_utf8_text_with_font(frame, text, left_top, font_path, font_size=20, text_color=(255, 255, 255)):
    """Draw UTF-8 text with custom font using Pillow."""
    # Convert OpenCV image to PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Initialize drawing context
    draw = ImageDraw.Draw(frame_pil)

    # Draw the text
    draw.text(left_top, text, font=font, fill=text_color)

    # Convert the PIL image back to OpenCV format
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    return frame

def draw_utf8_text_with_background(frame, text, left_top, font_path, font_size=20,
                                   text_color=(255, 255, 255), background_color=(0, 0, 0),
                                   background_alpha=0.5, background_padding=5):
    """Draw UTF-8 text with a background for better contrast using Pillow."""
    # Convert OpenCV image to PIL format
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Initialize drawing context
    draw = ImageDraw.Draw(frame_pil)

    # Calculate text size using textbbox (left, top, right, bottom)
    text_bbox = draw.textbbox(left_top, text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Define background rectangle
    background_rect = (
        left_top[0] - background_padding,
        left_top[1] - background_padding,
        left_top[0] + text_width + background_padding,
        left_top[1] + text_height + background_padding
    )

    # Draw background rectangle
    overlay = frame_pil.copy()
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(background_rect, fill=background_color)

    # Blend background with the original image
    frame_pil = Image.blend(frame_pil, overlay, alpha=background_alpha)

    # Draw text on top of background
    draw = ImageDraw.Draw(frame_pil)
    draw.text(left_top, text, font=font, fill=text_color)

    # Convert the PIL image back to OpenCV format
    frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

    return frame

if __name__ == "__main__":

    # Load the image
    frame = cv2.imread("/opt/images/spectrum2.png")
    if frame is None:
        print("Error: Image file not found or unable to load.")
        exit()

    # Path to the NotoSansJP-Light.ttf font file
    font_path = "fonts/NotoSansJP-Light.ttf"  # Specify your font path here

    # Draw UTF-8 text with custom font
    frame = draw_utf8_text_with_font(frame, "こんにちは, World!", (50, 100), font_path, font_size=30)

    # Draw UTF-8 text with background
    frame = draw_utf8_text_with_background(frame, "こんにちは, OpenCV!", (50, 200), font_path, font_size=30, background_alpha=0.5)

    # Resize the frame
    frame = cv2.resize(frame, (600, 600))

    # Show the frame with the added text
    cv2.imshow('frame', frame)
    cv2.waitKey(0)

    # Release resources
    cv2.destroyAllWindows()
