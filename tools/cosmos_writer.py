# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import omni
import omni.replicator.core as rep
import numpy as np
import json
from typing import Optional
import omni.timeline
import warp as wp
from omni.replicator.core import functional as F
from omni.replicator.core.annotators import AnnotatorRegistry
from omni.replicator.core.backends import io_queue
from omni.replicator.core.writers import Writer

try:
    from video_encoding import get_video_encoding_interface
except ImportError:
    pass # Handle missing extension gracefully if possible

__version__ = "0.0.1"

@wp.kernel
def rgb_to_grey_and_blur(
    data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array2d(dtype=wp.uint8)
):
    i, j = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]

    # Only process non-border pixels for the blur
    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        # Gaussian kernel 3x3
        kernel = wp.mat33f(1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0) / 16.0

        sum = 0.0
        # Apply convolution with RGB to grayscale conversion in a single pass
        for ki in range(-1, 2):
            for kj in range(-1, 2):
                # Convert RGB to grayscale using standard weights
                gray_val = (
                    wp.float(data_in[i + ki, j + kj, 0]) * 0.299  # Red
                    + wp.float(data_in[i + ki, j + kj, 1]) * 0.587  # Green
                    + wp.float(data_in[i + ki, j + kj, 2]) * 0.114  # Blue
                )
                sum += gray_val * kernel[ki + 1, kj + 1]

        data_out[i, j] = wp.uint8(wp.clamp(sum, 0.0, 255.0))
    else:
        # For border pixels, just convert to grayscale without blur
        data_out[i, j] = wp.uint8(
            wp.float(data_in[i, j, 0]) * 0.299
            + wp.float(data_in[i, j, 1]) * 0.587
            + wp.float(data_in[i, j, 2]) * 0.114
        )


@wp.kernel
def sobel_and_suppress(
    data_in: wp.array2d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    low_threshold: float,
    high_threshold: float,
):
    i, j = wp.tid()
    height = data_in.shape[0]
    width = data_in.shape[1]

    if i > 0 and i < height - 2 and j > 0 and j < width - 2:
        # Compute Sobel gradients with proper kernel weights
        gx = (
            -1.0 * wp.float(data_in[i - 1, j - 1])
            + -2.0 * wp.float(data_in[i, j - 1])
            + -1.0 * wp.float(data_in[i + 1, j - 1])
            + 1.0 * wp.float(data_in[i - 1, j + 1])
            + 2.0 * wp.float(data_in[i, j + 1])
            + 1.0 * wp.float(data_in[i + 1, j + 1])
        )

        gy = (
            -1.0 * wp.float(data_in[i - 1, j - 1])
            + -2.0 * wp.float(data_in[i - 1, j])
            + -1.0 * wp.float(data_in[i - 1, j + 1])
            + 1.0 * wp.float(data_in[i + 1, j - 1])
            + 2.0 * wp.float(data_in[i + 1, j])
            + 1.0 * wp.float(data_in[i + 1, j + 1])
        )

        # Compute gradient magnitude
        magnitude = wp.sqrt(gx * gx + gy * gy)

        # Calculate gradient direction and handle edge cases
        angle = wp.atan2(gy, gx) * 180.0 / 3.14159
        if angle < 0:
            angle += 180.0

        # Get interpolated magnitudes along gradient direction
        g00 = wp.float(0.0)  # First interpolated point
        g01 = wp.float(0.0)  # Second interpolated point
        xstep = wp.float(0.0)
        ystep = wp.float(0.0)

        # Determine interpolation direction based on angle
        if angle <= 22.5 or angle > 157.5:  # ~0 degrees
            xstep = wp.float(1.0)
            ystep = wp.float(0.0)
        elif angle > 22.5 and angle <= 67.5:  # ~45 degrees
            xstep = wp.float(1.0)
            ystep = wp.float(1.0)
        elif angle > 67.5 and angle <= 112.5:  # ~90 degrees
            xstep = wp.float(0.0)
            ystep = wp.float(1.0)
        else:  # ~135 degrees
            xstep = wp.float(-1.0)
            ystep = wp.float(1.0)

        # Interpolate gradient magnitudes
        # Forward direction
        x1 = wp.float(j) + xstep
        y1 = wp.float(i) + ystep
        if x1 >= 0 and x1 < width and y1 >= 0 and y1 < height:
            x1_floor = wp.int32(x1)
            y1_floor = wp.int32(y1)
            if (
                x1_floor >= 0
                and x1_floor < width - 1
                and y1_floor >= 0
                and y1_floor < height - 1
            ):
                gx1 = (
                    -1.0 * wp.float(data_in[y1_floor - 1, x1_floor - 1])
                    + -2.0 * wp.float(data_in[y1_floor, x1_floor - 1])
                    + -1.0 * wp.float(data_in[y1_floor + 1, x1_floor - 1])
                    + 1.0 * wp.float(data_in[y1_floor - 1, x1_floor + 1])
                    + 2.0 * wp.float(data_in[y1_floor, x1_floor + 1])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor + 1])
                )
                gy1 = (
                    -1.0 * wp.float(data_in[y1_floor - 1, x1_floor - 1])
                    + -2.0 * wp.float(data_in[y1_floor - 1, x1_floor])
                    + -1.0 * wp.float(data_in[y1_floor - 1, x1_floor + 1])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor - 1])
                    + 2.0 * wp.float(data_in[y1_floor + 1, x1_floor])
                    + 1.0 * wp.float(data_in[y1_floor + 1, x1_floor + 1])
                )
                g00 = wp.sqrt(gx1 * gx1 + gy1 * gy1)

        # Backward direction
        x2 = wp.float(j) - xstep
        y2 = wp.float(i) - ystep
        if x2 >= 0 and x2 < width and y2 >= 0 and y2 < height:
            x2_floor = wp.int32(x2)
            y2_floor = wp.int32(y2)
            if (
                x2_floor >= 0
                and x2_floor < width - 1
                and y2_floor >= 0
                and y2_floor < height - 1
            ):
                gx2 = (
                    -1.0 * wp.float(data_in[y2_floor - 1, x2_floor - 1])
                    + -2.0 * wp.float(data_in[y2_floor, x2_floor - 1])
                    + -1.0 * wp.float(data_in[y2_floor + 1, x2_floor - 1])
                    + 1.0 * wp.float(data_in[y2_floor - 1, x2_floor + 1])
                    + 2.0 * wp.float(data_in[y2_floor, x2_floor + 1])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor + 1])
                )
                gy2 = (
                    -1.0 * wp.float(data_in[y2_floor - 1, x2_floor - 1])
                    + -2.0 * wp.float(data_in[y2_floor - 1, x2_floor])
                    + -1.0 * wp.float(data_in[y2_floor - 1, x2_floor + 1])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor - 1])
                    + 2.0 * wp.float(data_in[y2_floor + 1, x2_floor])
                    + 1.0 * wp.float(data_in[y2_floor + 1, x2_floor + 1])
                )
                g01 = wp.sqrt(gx2 * gx2 + gy2 * gy2)

        # Strict non-maximum suppression with interpolation
        if magnitude > g00 and magnitude > g01:
            # Scale magnitude to match OpenCV's scaling
            scaled_magnitude = magnitude * 3.0

            if scaled_magnitude >= high_threshold:
                data_out[i, j, 0] = wp.uint8(255)  # Strong edge
            elif scaled_magnitude >= low_threshold:
                data_out[i, j, 0] = wp.uint8(127)  # Weak edge
            else:
                data_out[i, j, 0] = wp.uint8(0)  # Non-edge
        else:
            data_out[i, j, 0] = wp.uint8(0)
    else:
        data_out[i, j, 0] = wp.uint8(0)


@wp.kernel
def hysteresis_thresholding(data_inout: wp.array3d(dtype=wp.uint8)):
    i, j = wp.tid()
    height = data_inout.shape[0]
    width = data_inout.shape[1]

    if i > 0 and i < height - 1 and j > 0 and j < width - 1:
        # Only process weak edges
        if data_inout[i, j, 0] == 127:
            # Check 8-connected neighbors
            has_strong_neighbor = float(0.0)

            for di in range(-1, 2):
                if i + di < 0 or i + di >= height:
                    continue
                for dj in range(-1, 2):
                    if j + dj < 0 or j + dj >= width:
                        continue
                    if di == 0 and dj == 0:
                        continue
                    if data_inout[i + di, j + dj, 0] == 255:
                        has_strong_neighbor += 1.0
                        break
                if has_strong_neighbor >= 1.0:
                    break

            # Convert weak edge to strong if connected to strong edge, otherwise suppress
            if has_strong_neighbor >= 1.0:
                data_inout[i, j, 0] = wp.uint8(255)
            else:
                data_inout[i, j, 0] = wp.uint8(0)

    # Populate B and G channels for video encoder
    data_inout[i, j, 1] = data_inout[i, j, 0]
    data_inout[i, j, 2] = data_inout[i, j, 0]


@wp.kernel
def rand_colours(
    data_in: wp.array2d(dtype=wp.uint32), data_out: wp.array3d(dtype=wp.uint8)
):
    i, j = wp.tid()
    instance_id = data_in[i, j]

    # Check if this is background (index 0)
    if instance_id == 0:
        # Set background to black
        data_out[i, j, 0] = wp.uint8(0)
        data_out[i, j, 1] = wp.uint8(0)
        data_out[i, j, 2] = wp.uint8(0)

    # Convert instance_id to color using HSV with pastel parameters
    state_h = wp.rand_init(wp.int32(instance_id))
    state_s = wp.rand_init(wp.int32(instance_id), 1)
    state_v = wp.rand_init(wp.int32(instance_id), 2)

    # Pastel colors have moderate saturation and high value/brightness
    h = wp.randf(state_h)
    s = 0.3 + wp.randf(state_s) * 0.3  # Lower saturation (0.3-0.6) for pastel effect
    v = 0.9 + wp.randf(state_v) * 0.1  # High value/brightness (0.9-1.0)

    # HSV to RGB conversion
    K = wp.vec4f(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0)
    p = wp.vec3f(
        wp.abs(wp.frac(h + K.x) * 6.0 - K.w),
        wp.abs(wp.frac(h + K.y) * 6.0 - K.w),
        wp.abs(wp.frac(h + K.z) * 6.0 - K.w),
    )
    clamped = wp.vec3f(
        wp.clamp(p[0] - K.x, 0.0, 1.0),
        wp.clamp(p[1] - K.x, 0.0, 1.0),
        wp.clamp(p[2] - K.x, 0.0, 1.0),
    )
    rgb = v * wp.lerp(wp.vec3f(K.x), clamped, s)

    data_out[i, j, 0] = wp.uint8(rgb[0] * 255.0)
    data_out[i, j, 1] = wp.uint8(rgb[1] * 255.0)
    data_out[i, j, 2] = wp.uint8(rgb[2] * 255.0)

@wp.kernel
def color_semantic(
    data_in: wp.array2d(dtype=wp.uint32), 
    data_out: wp.array3d(dtype=wp.uint8),
    color_map: wp.array2d(dtype=wp.uint8)  # array de colores predefinidos
):
    i, j = wp.tid()
    class_id = data_in[i, j]

    for c in range(4):
        data_out[i, j, c] = color_map[class_id, c]


@wp.kernel
def shade_segmentation(
    segmentation: wp.array3d(dtype=wp.uint8),
    normals: wp.array3d(dtype=wp.float32),
    shading_out: wp.array3d(dtype=wp.uint8),
    light_source: wp.array(dtype=wp.vec3f),
):
    """Apply pastel-like colorization to semantic segmentation and shading using surface normals.

    Args:
        segmentation: Input semantic segmentation image with instance IDs (H,W)
        normals: Surface normal vectors (H,W,3)
        shading_out: Output colorized segmentation image (H,W,3)
        light_source: Position of light source
    """
    i, j = wp.tid()

    normal = normals[i, j]
    normals_normalized = wp.normalize(wp.vec3f(normal[0], normal[1], normal[2]))
    light_source_vec = wp.normalize(light_source[0])

    # Calculate base shading from dot product (ranges from -1 to 1)
    base_shade = wp.dot(normals_normalized, light_source_vec)
    # Remap from [-1, 1] to desired shading range [min_shade, 1.0]
    min_shade = 0.5  # Adjusted for pastel effect
    shade = wp.clamp(wp.lerp(min_shade, 1.0, (base_shade + 1.0) * 0.5), 0.0, 1.0)

    # Apply shading and convert to uint8
    shading_out[i, j, 0] = wp.uint8(wp.float(segmentation[i, j, 0]) * shade)
    shading_out[i, j, 1] = wp.uint8(wp.float(segmentation[i, j, 1]) * shade)
    shading_out[i, j, 2] = wp.uint8(wp.float(segmentation[i, j, 2]) * shade)


@wp.kernel
def colorize_depth(
    data_in: wp.array2d(dtype=wp.float32),
    data_out: wp.array3d(dtype=wp.uint8),
    near: float,
    far: float,
):
    """Apply colorization to depth data.

    Args:
        distance_data: Input depth data (H,W)
        output: Output colorized depth image (H,W,3)
        near: Minimum depth value to consider
        far: Maximum depth value to consider
    """
    i, j = wp.tid()

    # Get the depth value
    depth = data_in[i, j]

    # Skip invalid values
    if depth != wp.inf and depth != -wp.inf:
        # Clip depth to range [near, far]
        clipped_depth = wp.clamp(depth, near, far) + 1e-5

        # Apply log normalization
        normalized = (wp.log(clipped_depth) - wp.log(near)) / (
            wp.log(far) - wp.log(near)
        )

        # Invert and scale to 0-255 range
        color_value = wp.uint8((1.0 - normalized) * 255.0)

        # Set RGB channels to the same value (grayscale)
        data_out[i, j, 0] = color_value
        data_out[i, j, 1] = color_value
        data_out[i, j, 2] = color_value
    else:
        # For invalid depth values, set to black
        data_out[i, j, 0] = wp.uint8(0)
        data_out[i, j, 1] = wp.uint8(0)
        data_out[i, j, 2] = wp.uint8(0)
    # Set alpha channel to 0
    data_out[i, j, 3] = wp.uint8(255)


class CosmosWriter(Writer):
    """Writer class for generating input to Cosmos Transfer1.

    This writer generates videos for various modalities that can be used as input to Cosmos Transfer1. The included
    modalities are:
    - RGB
    - Shaded Instance Segmentation
    - Instance Segmentation
    - Distance to camera (Depth)
    - Edges

    If using a ``trigger.on_time`` node, the writer will automatically increment the clip index when the trigger fires.
    Otherwise, the clip index can be incremented manually by calling the ``next_clip`` method.

    Args:
        backend: The backend to use for writing the video.
        video_filepath: Path where the output video will be saved (e.g. "/home/user/Videos/my_video.mp4").
        segmentation_mapping: An optional dictionary mapping semantic labels to specific colors.
        use_instance_id: Whether to use instance id segmentation instead of instance segmentation. Instance ID
            segmentation does not require assets be semantically annotated.
        canny_threshold_low: The lower threshold for the Canny edge detector.
        canny_threshold_high: The higher threshold for the Canny edge detector.
    """

    def __init__(
        self,
        backend,
        #segmentation_mapping: Optional[dict] = None, # { "(255, 0, 0, 255)": {"class": "Worker"},}
        segmentation_mapping: Optional[dict] = None,
        use_instance_id: bool = True,
        canny_threshold_low: int = 10,
        canny_threshold_high: int = 100,
        video_fps: int = 30,  # FPS for output video (lower = slower playback)
    ):
        self._backend = backend
        self.version = __version__

        semantic_params = {"colorize": True}
        if segmentation_mapping:
            semantic_params["mapping"] = json.dumps(segmentation_mapping)

        segmentation_annotator = (
            "semantic_segmentation"
            if use_instance_id
            else "semantic_segmentation"
        )
        self.annotators = [
            AnnotatorRegistry.get_annotator(
                segmentation_annotator, init_params=semantic_params, device="cuda"
            ),
            AnnotatorRegistry.get_annotator("normals", device="cuda"),
            AnnotatorRegistry.get_annotator("distance_to_camera").augment(
                colorize_depth,
                name="depth",
                near=0.1,
                far=100,
                data_out_shape=(-1, -1, 4),
            ),
            AnnotatorRegistry.get_annotator("rgb", device="cuda"),
        ]

        self._canny_threshold_low = canny_threshold_low
        self._canny_threshold_high = canny_threshold_high
        self._frame_id = 0
        self._clip_idx = 0
        self._frame_rate = None
        self._video_fps = video_fps  # FPS for output video
        self._light_source = None
        self._cached_buffers = {}
        # Manage per-episode folders
        self._base_output_dir = backend.output_dir
        self._episode_subdir = ""

    def set_episode_output(self, episode_dir: str):
        """Select output subdirectory for current episode (relative to base output root)."""
        if episode_dir.startswith(self._base_output_dir):
            rel_path = os.path.relpath(episode_dir, self._base_output_dir)
        else:
            rel_path = episode_dir
        # Normalize to avoid '../'
        rel_path = rel_path.strip().strip("./")
        self._episode_subdir = rel_path

    def _episode_path(self, relative_path: str) -> str:
        if self._episode_subdir:
            return f"{self._episode_subdir}/{relative_path}"
        return relative_path

    def _get_shaded_segmentation(self, normals, segmentation):
        """Get shaded segmentation from the segmentation and normals.

        Args:
            normals: Normals array (H,W,3)
            segmentation: Segmentation array (H,W)
        """
        height, width = segmentation.shape[:2]
        device = segmentation.device
        if self._cached_buffers.get("light_source") is None:
            self._cached_buffers["light_source"] = wp.array(
                [0.0, 0.0, 1.0], dtype=wp.vec3f, device=device
            )
        light_source = self._cached_buffers["light_source"]
        shaded_segmentation_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 3),
            device=device,
            owner=False,
            requires_grad=False,
        )
        wp.launch(
            kernel=shade_segmentation,
            dim=(height, width),
            inputs=[segmentation, normals, shaded_segmentation_out, light_source],
            device=device,
        )
        return shaded_segmentation_out

    def _get_segmentation(self, semantic_segmentation):
        """Get segmentation from the instance segmentation.

        Args:
            semantic_segmentation: Instance segmentation array (H,W,C)
        """#
        colors_list ={
            '0': (128,0,0,255),
            '1': (0,255,0,255),
            '2': (0,0,255,255),
            '3': (255,0,0,255),
            '4': (0,0,255,255),
            '5': (0,255,0,255),
            '6': (128,128,128,255),
            '7': (0,255,0,255),
            '8': (0,255,0,255),
            '9': (0,0,128,255),
            '10': (0,255,0,255),
            '11': (255,0,0,255),
            '12': (255,0,0,255),
            '13': (255,0,0,255),
            '14': (255,0,0,255),
            '15': (32,32,32,255),
            '16': (255,0,0,255)
        }
        self.colors_list = colors_list
        
        class_colors_array = wp.array(np.array(list(colors_list.values()), dtype=np.uint8))
        height, width = semantic_segmentation.shape[:2]
        device = semantic_segmentation.device
        segmentation_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 4),
            device=device,
        )
        wp.launch(
            kernel=color_semantic,
            dim=(height, width),
            inputs=[semantic_segmentation, segmentation_out, class_colors_array],
            device=device,
        )
        
        return segmentation_out

    def _get_canny_edges(self, shaded_segmentation):
        """Get canny edges from the shaded segmentation.

        Args:
            shaded_segmentation: Shaded segmentation array (H,W,3)
        """
        height, width = shaded_segmentation.shape[:2]
        device = shaded_segmentation.device

        # Allocate output buffer
        # Encoder expects 3 channels
        canny_edges_out = wp.empty(
            dtype=wp.uint8,
            shape=(height, width, 3),
            device=device,
        )

        # Cache this buffer to avoid reallocating it on each call
        if self._cached_buffers.get("greyscale") is None:
            self._cached_buffers["greyscale"] = wp.empty(
                dtype=wp.uint8,
                shape=(height, width),
                device=device,
                owner=False,
                requires_grad=False,
            )

        # Step 1: Convert to greyscale and blur
        wp.launch(
            kernel=rgb_to_grey_and_blur,
            dim=(height, width),
            inputs=[shaded_segmentation, self._cached_buffers["greyscale"]],
            device=device,
        )

        # Step 2: Apply sobel operator
        wp.launch(
            kernel=sobel_and_suppress,
            dim=(height, width),
            inputs=[
                self._cached_buffers["greyscale"],
                canny_edges_out,
                self._canny_threshold_low,
                self._canny_threshold_high,
            ],
            device=device,
        )

        # Step 3: Apply hysteresis thresholding
        wp.launch(
            kernel=hysteresis_thresholding,
            dim=(height, width),
            inputs=[canny_edges_out],
            device=device,
        )

        return canny_edges_out

    def _get_semantic_json(self, semantic_segmentation):
        semantic_annotator = self.annotators[0]
        data = semantic_annotator.get_data()['info']['idToLabels']
       
        return data

    def write(self, data):
        """Write video data optimized for Cosmos Transfer1.

        Args:
            data: Dictionary containing frame data with keys:
                - rgb: RGB image array (H,W,3)
                - semantic_segmentation_fast: Instance segmentation image array (H,W,C)
                - shaded_semantic_segmentation: Shaded instance segmentation image array (H,W,C)
                - depth: Depth image array (H,W,1)
                - edges: Edge image array (H,W,1)
        """
        # Only write if explicitly active (logic moved to orchestrator step in sim_main)
        # But we can also control clip index here
        
        sequence_id = 0
        if "trigger_outputs" in data and data["trigger_outputs"]:
            for trigger_name, call_count in data["trigger_outputs"].items():
                if "on_time" in trigger_name:
                    sequence_id = call_count
        
        if sequence_id != self._clip_idx:
            self.next_clip()

        if self._frame_rate is None:
            timeline_iface = omni.timeline.get_timeline_interface()
            self._frame_rate = timeline_iface.get_time_codes_per_seconds()
        
        semantic_segmentation = data["semantic_segmentation"]["data"]
        #segmentation = self._get_segmentation(semantic_segmentation)
        segmentation_label = self._get_semantic_json(semantic_segmentation)
        normals = data["normals"]
        depth = data["depth"]
        rgb = data["rgb"]
        shaded_seg = self._get_shaded_segmentation(normals, semantic_segmentation)
        edges = self._get_canny_edges(shaded_seg)
        semantic_annotator = self.annotators[0]
        data = semantic_annotator.get_data()
        # DEBUG: Check if we have data
        # if self._frame_id % 60 == 0:
        #    print(f"[CosmosWriter] Writing frame {self._frame_id}. RGB shape: {rgb.shape}, Seg data size: {semantic_segmentation.size}")

        self._backend.schedule(
            F.write_image,
            data=rgb,
            path=self._episode_path(f"clip_{self._clip_idx:04}/rgb/rgb_{self._frame_id:04}.png"),
        )
        self._backend.schedule(
            F.write_image,
            data=shaded_seg,
            path=self._episode_path(f"clip_{self._clip_idx:04}/shaded_seg/shaded_seg_{self._frame_id:04}.png"),
        )

        self._backend.schedule(
            F.write_image,
            data=semantic_segmentation,
            path=self._episode_path(f"clip_{self._clip_idx:04}/semantic_segmentation/semantic_segmentation_{self._frame_id:04}.png"),
        )
        self._backend.schedule(
            F.write_image,
            data=depth,
            path=self._episode_path(f"clip_{self._clip_idx:04}/depth/depth_{self._frame_id:04}.png"),
        )
        self._backend.schedule(
            F.write_image,
            data=edges,
            path=self._episode_path(f"clip_{self._clip_idx:04}/edges/edges_{self._frame_id:04}.png"),
        )
        self._backend.schedule(
            F.write_json,
            data=segmentation_label,
            path=self._episode_path(f"clip_{self._clip_idx:04}/semantic_segmentation/segmentation_label/segmentation_label_{self._frame_id:04}.json"),
        )
        self._frame_id += 1

    def on_final_frame(self):
        import subprocess  # Import at function start to avoid scope issues
        import sys
        
        if self._frame_id == 0:
            print(f"[CosmosWriter] on_final_frame() called but _frame_id=0, skipping")
            return

        print(f"[CosmosWriter] on_final_frame() iniciado para {self._frame_id} frames", flush=True)
        print(f"[CosmosWriter] Esperando a que terminen de escribirse las imágenes...", flush=True)
        sys.stdout.flush()
        
        io_queue.wait_until_done()
        
        print(f"[CosmosWriter] ✓ Imágenes escritas, generando videos...", flush=True)
        sys.stdout.flush()
        
        output_dir = self._base_output_dir
        if self._episode_subdir:
            output_dir = os.path.join(output_dir, self._episode_subdir)
        clip_dir = f"{output_dir}/clip_{self._clip_idx:04}"
        fps = self._video_fps  # Use configured video FPS (default 15)
        
        print(f"[CosmosWriter] Finalizing clip {self._clip_idx} with {self._frame_id} frames at {fps} FPS...", flush=True)
        print(f"[CosmosWriter] Clip dir: {clip_dir}", flush=True)
        sys.stdout.flush()

        # Try NVIDIA Video Encoding first
        encoding_success = False
        try:
            video_encoding = get_video_encoding_interface()
            if video_encoding:
                print("[CosmosWriter] Using NVIDIA video_encoding...")
                for key in ["rgb", "semantic_segmentation", "edges", "depth", "shaded_seg"]:
                    video_encoding.start_encoding(
                        video_filename=f"{clip_dir}/{key}.mp4",
                        framerate=fps,
                        nframes=self._frame_id,
                        overwrite_video=True,
                    )
                    for i in range(self._frame_id):
                        path = f"{clip_dir}/{key}/{key}_{i:04}.png"
                        video_encoding.encode_next_frame_from_file(path)
                    video_encoding.finalize_encoding()
                encoding_success = True
        except Exception as e:
            print(f"[CosmosWriter] NVIDIA video_encoding failed/missing: {e}")

        # Fallback to FFmpeg if NVIDIA encoding failed
        if not encoding_success:
            print("[CosmosWriter] Attempting fallback to FFmpeg...")
            
            # Check if ffmpeg is installed
            try:
                subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                print("[CosmosWriter] Error: FFmpeg is not installed or not in PATH. Cannot generate videos.")
                return

            for key in ["rgb", "semantic_segmentation", "edges", "depth", "shaded_seg"]:
                input_pattern = f"{clip_dir}/{key}/{key}_%04d.png"
                output_video = f"{clip_dir}/{key}.mp4"
                
                # FFmpeg command:
                # -y: overwrite
                # -framerate: input fps
                # -i: input pattern
                # -c:v libx264: video codec
                # -pix_fmt yuv420p: pixel format for compatibility
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(fps),
                    "-i", input_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    output_video
                ]
                
                try:
                    # Run quietly
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    print(f"[CosmosWriter] Generated video: {output_video}")
                except subprocess.CalledProcessError as e:
                    print(f"[CosmosWriter] FFmpeg failed for {key}: {e}")

        self._frame_id = 0

    def __del__(self):
        # Ensure videos are written if object is destroyed
        self.on_final_frame()

    def next_clip(self):
        """Finalize current clip and update parameters for the next one.

        - Combines generated frames into videos
        - Resets frame counter
        - Increments output directory
        """
        self.on_final_frame()
        self._clip_idx += 1

