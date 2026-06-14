"""
Pure ONNX Runtime inference for YOLOv8n car damage detection.

Replaces the Ultralytics/PyTorch runtime in production while preserving
the same confidence threshold, input size, letterbox preprocessing, and NMS
behavior used by the original .pt pipeline.
"""

from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort


# Matches Ultralytics predict defaults for this project.
DEFAULT_IOU_THRESHOLD = 0.7
LETTERBOX_COLOR = (114, 114, 114)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert center-format boxes to corner format."""

    converted = np.empty_like(boxes)
    converted[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    converted[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    converted[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    converted[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return converted


def clip_boxes(
    boxes: np.ndarray,
    image_shape: tuple[int, int]
) -> np.ndarray:
    """Clip boxes to image bounds using (height, width) shape."""

    height, width = image_shape
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, width)
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, height)
    return boxes


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    input_shape: tuple[int, int],
    original_shape: tuple[int, int]
) -> np.ndarray:
    """
    Rescale letterboxed model coordinates back to the original image.

    Mirrors ultralytics.utils.ops.scale_boxes() when ratio_pad is derived
    from the input and original image shapes.
    """

    if boxes_xyxy.size == 0:
        return boxes_xyxy

    scaled = boxes_xyxy.copy()
    gain = min(
        input_shape[0] / original_shape[0],
        input_shape[1] / original_shape[1]
    )
    pad_x = round(
        (input_shape[1] - round(original_shape[1] * gain)) / 2 - 0.1
    )
    pad_y = round(
        (input_shape[0] - round(original_shape[0] * gain)) / 2 - 0.1
    )

    scaled[:, [0, 2]] -= pad_x
    scaled[:, [1, 3]] -= pad_y
    scaled[:, :4] /= gain
    return clip_boxes(scaled, original_shape)


def non_max_suppression(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int = 300
) -> list[int]:
    """
    Greedy NMS aligned with torchvision/Ultralytics behavior.
    Returns indices of boxes to keep.
    """

    if boxes_xyxy.size == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0 and len(keep) < max_detections:
        current = order[0]
        keep.append(int(current))

        if order.size == 1:
            break

        remaining = order[1:]
        current_box = boxes_xyxy[current]
        others = boxes_xyxy[remaining]

        inter_x1 = np.maximum(current_box[0], others[:, 0])
        inter_y1 = np.maximum(current_box[1], others[:, 1])
        inter_x2 = np.minimum(current_box[2], others[:, 2])
        inter_y2 = np.minimum(current_box[3], others[:, 3])

        inter_w = np.maximum(0.0, inter_x2 - inter_x1)
        inter_h = np.maximum(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        if inter_area.sum() == 0:
            order = remaining
            continue

        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
        union = current_area + other_area - inter_area
        iou = inter_area / union
        order = remaining[iou <= iou_threshold]

    return keep


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    YOLOv8 letterbox preprocessing.

    Resize while preserving aspect ratio, then pad to imgsz x imgsz.
    This mirrors Ultralytics LetterBox(auto=False, center=True).
    """

    height, width = image.shape[:2]
    target_height, target_width = new_shape

    scale = min(
        target_width / width,
        target_height / height
    )
    resized_width = round(width * scale)
    resized_height = round(height * scale)

    if (width, height) != (resized_width, resized_height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )

    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    pad_left = pad_width / 2
    pad_top = pad_height / 2

    top = round(pad_top - 0.1)
    bottom = round(pad_top + 0.1)
    left = round(pad_left - 0.1)
    right = round(pad_left + 0.1)

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_COLOR
    )

    return image


class YoloOnnxDetector:
    """YOLOv8 ONNX detector using onnxruntime only."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.15,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        imgsz: int = 640,
        class_name: str = "Damage"
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.class_name = class_name

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Letterbox to 640x640 and build the ONNX input tensor.

        Ultralytics converts OpenCV BGR images to RGB before inference.
        Output tensor shape: (1, 3, 640, 640), float32, normalized to [0, 1].
        """

        letterboxed = letterbox(
            image,
            (self.imgsz, self.imgsz)
        )

        # Match Ultralytics predictor preprocessing: BGR -> RGB.
        rgb_image = letterboxed[..., ::-1]
        tensor = rgb_image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return tensor

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Decode YOLOv8 ONNX output, apply confidence filtering and NMS,
        then map boxes back to the original image coordinate space.

        Exported model output shape: (1, 4 + num_classes, num_predictions)
        For this project: (1, 5, 8400) with one class named Damage.
        """

        prediction = np.squeeze(output, axis=0)
        num_classes = prediction.shape[0] - 4
        class_scores = prediction[4:4 + num_classes]
        candidate_mask = class_scores.max(axis=0) > self.conf_threshold

        prediction = prediction[:, candidate_mask].T
        if prediction.size == 0:
            return np.empty((0, 4)), np.empty((0,))

        boxes_xyxy = xywh_to_xyxy(prediction[:, :4])
        confidences = prediction[:, 4:4 + num_classes].max(axis=1)

        keep_indices = non_max_suppression(
            boxes_xyxy,
            confidences,
            self.iou_threshold
        )

        boxes_xyxy = boxes_xyxy[keep_indices]
        confidences = confidences[keep_indices]
        boxes_xyxy = scale_boxes_to_original(
            boxes_xyxy,
            (self.imgsz, self.imgsz),
            original_shape
        )

        sort_order = confidences.argsort()[::-1]
        return boxes_xyxy[sort_order], confidences[sort_order]

    def draw_detections(
        self,
        image: np.ndarray,
        boxes_xyxy: np.ndarray,
        confidences: np.ndarray
    ) -> np.ndarray:
        """Draw bounding boxes and confidence labels with OpenCV."""

        annotated = image.copy()

        for box, confidence in zip(boxes_xyxy, confidences):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.class_name} {confidence:.2f}"

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        return annotated

    def detect(
        self,
        image_path: str
    ) -> tuple[list[float], np.ndarray]:
        """
        Run full ONNX inference on an image path.

        Returns:
            confidences: sorted detection confidence scores
            annotated_image: original image with boxes drawn
        """

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")

        original_shape = image.shape[:2]

        # Preprocess: letterbox + BGR->RGB + tensor build.
        input_tensor = self.preprocess(image)

        # Inference: ONNX Runtime session execution.
        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )

        # Postprocess: decode, threshold, NMS, rescale to original image.
        boxes_xyxy, confidences = self.postprocess(
            outputs[0],
            original_shape
        )

        annotated_image = self.draw_detections(
            image,
            boxes_xyxy,
            confidences
        )

        return confidences.tolist(), annotated_image
