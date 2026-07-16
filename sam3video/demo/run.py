import argparse
import sys
import cv2

sys.path.insert(0, "..")
from sam3video.segmenter import VideoSegmenter, read_all_frames, video_frame_count_and_fps
from sam3video.rendering import overlay_masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True, help='text prompt, e.g. "person"')
    parser.add_argument("--output", default="segmented.mp4")
    parser.add_argument("--version", default="sam3.1", choices=["sam3", "sam3.1"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    width, height, fps = video_frame_count_and_fps(args.video)
    frames = read_all_frames(args.video)

    segmenter = VideoSegmenter(
        version=args.version,
        checkpoint_path=args.checkpoint,
        compile_model=args.compile,
    )

    rendered = list(frames)
    n_masked_frames = 0

    for frame_index, obj_ids, binary_masks in segmenter.segment(
        args.video, args.prompt, prompt_frame_index=args.prompt_frame
    ):
        if frame_index >= len(rendered):
            continue
        rendered[frame_index] = overlay_masks(frames[frame_index], obj_ids, binary_masks)
        n_masked_frames += 1

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {args.output}")

    try:
        for frame in rendered:
            writer.write(frame)
    finally:
        writer.release()

    print(f"Prompt: {args.prompt!r}")
    print(f"Frames with segmentation: {n_masked_frames}/{len(frames)}")
    print(f"Wrote segmented video to {args.output}")


if __name__ == "__main__":
    main()
