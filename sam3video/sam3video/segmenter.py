import os
import cv2
import numpy as np

from sam3.model_builder import build_sam3_predictor


class VideoSegmenter:
    def __init__(self, version="sam3.1", checkpoint_path=None, compile_model=False):
        self.predictor = build_sam3_predictor(
            version=version,
            checkpoint_path=checkpoint_path,
            compile=compile_model,
        )

    def segment(self, video_path, text_prompt, prompt_frame_index=0):
        start_response = self.predictor.handle_request(
            request=dict(type="start_session", resource_path=video_path)
        )
        session_id = start_response["session_id"]

        self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=prompt_frame_index,
                text=text_prompt,
            )
        )

        for step in self.predictor.handle_stream_request(
            request=dict(type="propagate_in_video", session_id=session_id)
        ):
            frame_index = step["frame_index"]
            outputs = step["outputs"]
            obj_ids = outputs["out_obj_ids"]
            binary_masks = outputs["out_binary_masks"]
            yield frame_index, obj_ids, binary_masks

        self.predictor.handle_request(request=dict(type="close_session", session_id=session_id))


def video_frame_count_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height, fps


def read_all_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames
