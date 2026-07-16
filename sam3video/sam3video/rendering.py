import numpy as np
import cv2

PALETTE = [
    (255, 99, 71),
    (60, 179, 113),
    (65, 105, 225),
    (238, 130, 238),
    (255, 215, 0),
    (0, 206, 209),
    (255, 105, 180),
    (154, 205, 50),
]


def overlay_masks(frame_bgr, obj_ids, binary_masks, alpha=0.5):
    overlay = frame_bgr.copy()
    for obj_id, mask in zip(obj_ids, binary_masks):
        color = PALETTE[int(obj_id) % len(PALETTE)]
        colored = np.zeros_like(frame_bgr)
        colored[mask] = color
        overlay = np.where(mask[..., None], cv2.addWeighted(overlay, 1 - alpha, colored, alpha, 0), overlay)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        ys, xs = np.where(mask)
        if len(xs) > 0:
            label_pos = (int(xs.min()), max(int(ys.min()) - 8, 12))
            cv2.putText(overlay, f"id {int(obj_id)}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return overlay
