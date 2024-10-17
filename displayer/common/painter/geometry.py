import cv2

def draw_box(image, lx, ly, rx, ry, label="", color=(0, 255, 0)):

    # Draw the bounding box
    cv2.rectangle(image, (lx, ly), (rx, ry), color, 2)

    # Draw the label text
    if label != "":
        # Get text size for background size
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw a filled rectangle as background for text
        cv2.rectangle(image, (lx, ly - text_size[1] - 5), (lx + text_size[0], ly), color, cv2.FILLED)

        # Put the label text on the image
        cv2.putText(image, label, (lx, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def draw_point(image, x, y, label="", color=(0, 255, 0)):

    # Draw the point
    cv2.circle(image, (x, y), 5, color, -1)

    # Draw the label text
    if label != "":
        # Put the label text on the image
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image


def draw_circle(image, x, y, radius, label="", color=(0, 255, 0)):

    # Draw the circle
    cv2.circle(image, (x, y), int(radius), color, 2)

    # Draw the label text
    if label != "":
        # Put the label text on the image
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return image