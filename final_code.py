"""
================================================================================
BLOOD LAYER SEGMENTATION - U-NET TRAINING PIPELINE
================================================================================
This consolidated script handles the complete training pipeline for automated
blood layer segmentation using U-Net architecture with MobileNet preprocessing.

OVERVIEW:
  Blood samples in centrifuge tubes separate into distinct layers:
    - Background  (class 0) – tube wall / non-blood region
    - Plasma       (class 1) – pale yellow top layer
    - Buffy Coat   (class 2) – thin white layer of WBCs/platelets
    - Red Blood Cells (class 3) – dense red bottom layer

  Pipeline stages:
    1. Load a frozen MobileNet SSD to detect the tube and find its vertical
       center (the buffy-coat transition region).
    2. Crop a fixed-size window (368×368 px) around that center.
    3. Feed cropped image + mask pairs to the U-Net in mini-batches.
    4. Save the best checkpoint and final model in SavedModel + H5 formats.

FIXES APPLIED:
  1. load_graph_def: replaced bare `tf.GraphDef()` (does not exist in TF2)
     with the correct `tf.compat.v1.GraphDef()`.
  2. detection_boxes indexing: clarified that the TF OD API returns boxes in
     [ymin, xmin, ymax, xmax] order; box[1] (xmin) and box[3] (xmax) are
     unused for the vertical-crop calculation.
  3. boxes_out iteration: `detection_boxes` has shape (batch, N_detections, 4).
     The original code iterated over every detection across the whole batch,
     producing N_detections center values instead of one per image.  Fixed to
     take only the highest-scoring (index-0) detection per image.
  4. Mask interpolation: ImageDataGenerator defaults to bilinear resampling,
     which corrupts discrete label pixel values (85, 170, 255). Fixed by
     passing `interpolation='nearest'` to the mask flow.
  5. img_dims width mismatch: `img_dims = (2048, 366)` did not match
     `crop_width = 368`, causing pad_images to add 2 spurious columns.
     Corrected to `(2048, 368)`.
  6. Bottom-edge crop clamping: the original code clamped `offset_height`
     to ≥ 0 (top overflow) but not to ≤ img_height - crop_height (bottom
     overflow). Added the missing lower clamp in both normalize_inputs and
     normalize_labels.
  7. ModelCheckpoint path extension: TF 2.x+ requires `.weights.h5` when
     `save_weights_only=True`.  Using `.ckpt` raises a warning and may fail
     on newer Keras versions.
  8. Batch-level exception handling: wrapped the per-batch processing in its
     own try/except so a single corrupt sample skips the batch rather than
     killing the entire generator.

Author: Michael Pan (UW-Madison)
Date: April 2026

DEPENDENCIES:
  - TensorFlow 2.13+
  - Keras (bundled with TF)
  - NumPy
  - Matplotlib
  - Custom modules: Utilities.image_utils, Utilities.file_io

USAGE:
    python unet_training_consolidated.py --train_dir /path/to/train \\
                                         --test_dir  /path/to/test  \\
                                         --epochs 100
================================================================================
"""

import os
import math
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras imports
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D,
    concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Custom utility modules (must be on PYTHONPATH)
from Utilities import image_utils, file_io

# Suppress noisy TF warnings during development; remove in production
warnings.filterwarnings('ignore')


# ==============================================================================
# CONSTANTS
# ==============================================================================

# Segmentation classes (pixel value → class index mapping in mask images)
CLASS_BACKGROUND    = 0   # pixel value   0 → class 0
CLASS_PLASMA        = 1   # pixel value  85 → class 1
CLASS_BUFFY_COAT    = 2   # pixel value 170 → class 2
CLASS_RED_BLOOD     = 3   # pixel value 255 → class 3

# Canonical crop dimensions used everywhere in the pipeline
CROP_HEIGHT = 368
CROP_WIDTH  = 368

# Height to which images are padded before the center crop
PAD_HEIGHT = 2048


# ==============================================================================
# U-NET MODEL ARCHITECTURE
# ==============================================================================

def build_unet_model(pretrained_weights=None, input_size=(368, 368, 3)):
    """
    Construct and compile a U-Net encoder-decoder for semantic segmentation.

    Architecture (Ronneberger et al., 2015):
    ┌─────────────────────────────────────────────────────────────────────┐
    │  ENCODER (downsampling)  →  BOTTLENECK  →  DECODER (upsampling)    │
    │  Skip connections concatenate encoder feature maps into the decoder │
    │  at each resolution level to preserve fine spatial detail.          │
    └─────────────────────────────────────────────────────────────────────┘

    Spatial resolution track (height dimension):
        Input: 368 → pool1: 184 → pool2: 92 → pool3: 46 → pool4: 23
        (bottleneck at 23×23×1024)
        up6: 46 → up7: 92 → up8: 184 → up9: 368
        Output: 368×368×4  (one logit channel per class)

    Design choices:
      - He normal weight init   : optimal for ReLU activations
      - Dropout 0.5 at levels 4 & 5 : regularises the deepest, richest features
      - softmax output          : yields per-pixel class probability distribution
      - sparse_categorical_crossentropy : expects integer class labels (not one-hot)

    Args:
        pretrained_weights (str | None): Path to a .h5 weights file to
            fine-tune from.  Pass None to train from scratch.
        input_size (tuple): (height, width, channels).  Default (368, 368, 3).

    Returns:
        tf.keras.Model: Compiled model ready for model.fit().
    """

    # ── ENCODER ────────────────────────────────────────────────────────────

    inputs = Input(input_size)

    # Level 1 — 64 filters; output spatial size = input (368×368)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # → 184×184

    # Level 2 — 128 filters
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # → 92×92

    # Level 3 — 256 filters
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # → 46×46

    # Level 4 — 512 filters + dropout before pooling
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)  # → 23×23

    # ── BOTTLENECK ─────────────────────────────────────────────────────────
    # Deepest representation: 23×23 spatial, 1024 feature channels
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    # ── DECODER ────────────────────────────────────────────────────────────
    # Each decoder level: upsample → conv → skip-connect → 2× conv block

    # Decoder Level 1 (23 → 46)
    up6    = Conv2D(512, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)   # skip from encoder level 4
    conv6  = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge6)
    conv6  = Conv2D(512, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv6)

    # Decoder Level 2 (46 → 92)
    up7    = Conv2D(256, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)   # skip from encoder level 3
    conv7  = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge7)
    conv7  = Conv2D(256, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv7)

    # Decoder Level 3 (92 → 184)
    up8    = Conv2D(128, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)   # skip from encoder level 2
    conv8  = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge8)
    conv8  = Conv2D(128, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv8)

    # Decoder Level 4 (184 → 368)
    up9    = Conv2D(64, 2, activation='relu', padding='same',
                    kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)   # skip from encoder level 1
    conv9  = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(merge9)
    conv9  = Conv2D(64, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv9)

    # ── OUTPUT ─────────────────────────────────────────────────────────────
    # 1×1 conv collapses 64 feature maps → 4 class probability channels
    conv10 = Conv2D(4, 1, activation='softmax')(conv9)

    # ── COMPILE ────────────────────────────────────────────────────────────
    model = Model(inputs=inputs, outputs=conv10)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',   # expects integer label maps
        metrics=['sparse_categorical_accuracy']
    )

    if pretrained_weights is not None:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        model.load_weights(pretrained_weights)

    return model


# ==============================================================================
# DATA PREPROCESSING HELPERS
# ==============================================================================

def normalize_inputs(images, center_y, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT):
    """
    Normalize a batch of raw RGB images and crop a window around the tube center.

    Processing per image:
      1. Cast to float32 and scale pixels to [0, 1].
      2. Ensure a channel dimension exists (handles accidental 2-D inputs).
      3. Pad height to PAD_HEIGHT so the crop window always has room.
      4. Compute a crop offset centered on `center_y`, clamped to valid range.
      5. Crop to (crop_height × crop_width).
      6. Stack into a batch tensor of shape (N, H, W, C).

    Args:
        images   (list | np.ndarray): Raw uint8 images, shape (H, W, 3) each.
        center_y (list[int])         : Pixel row of the tube center per image.
        crop_width  (int)            : Output crop width in pixels.
        crop_height (int)            : Output crop height in pixels.

    Returns:
        np.ndarray: Float32 batch of shape (N, crop_height, crop_width, 3).
    """
    normalized_images = []

    for img, center in zip(images, center_y):
        img = img.astype(np.float32)
        img = img / 255.0  # scale to [0, 1]

        # Guarantee (H, W, C) — flow_from_directory should always give 3-D
        # arrays, but guard against accidental 2-D greyscale leaking through
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        # Pad so the crop window never exceeds the image boundary at the top
        img = image_utils.pad_images(img, target_height=PAD_HEIGHT,
                                     target_width=crop_width)

        # Clamp offset so the crop stays within [0, PAD_HEIGHT - crop_height]
        # FIX: original code only clamped the top edge (offset >= 0); without
        # the upper clamp a center near the bottom would produce an out-of-
        # bounds crop in pad_images / crop_to_bounding_box.
        max_offset = PAD_HEIGHT - crop_height
        offset_height = int(np.clip(center - crop_height // 2, 0, max_offset))

        img = image_utils.crop_to_bounding_box(
            images=img,
            offset_height=offset_height,
            offset_width=0,
            crop_height=crop_height,
            crop_width=crop_width
        )

        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)

        normalized_images.append(np.expand_dims(img, axis=0))  # (1, H, W, C)

    return np.concatenate(normalized_images, axis=0)  # (N, H, W, C)


def normalize_labels(labels, center_y, crop_width=CROP_WIDTH, crop_height=CROP_HEIGHT):
    """
    Convert raw grayscale mask images to integer class-index arrays and crop.

    Mask pixel-value → class-index mapping:
        0   → 0  (Background)
        85  → 1  (Plasma)
        170 → 2  (Buffy Coat)
        255 → 3  (Red Blood Cells)

    NOTE: The ImageDataGenerator feeding this function must use
    `interpolation='nearest'` (see model_data_generator) so that discrete
    pixel values survive resizing without blending into intermediate values.

    Args:
        labels   (list | np.ndarray): Grayscale mask images, shape (H, W) each.
        center_y (list[int])         : Pixel row of the tube center per image.
        crop_width  (int)            : Output crop width in pixels.
        crop_height (int)            : Output crop height in pixels.

    Returns:
        np.ndarray: int32 batch of shape (N, crop_height, crop_width).
    """
    labels = np.array(labels)  # (N, H, W)  — collapse list to array first

    # Remap grayscale values to class indices in-place
    labels[labels == 85]  = CLASS_PLASMA       # 85  → 1
    labels[labels == 170] = CLASS_BUFFY_COAT   # 170 → 2
    labels[labels == 255] = CLASS_RED_BLOOD    # 255 → 3
    # 0 already maps to CLASS_BACKGROUND (0), no action needed

    normalized_labels = []

    for label, center in zip(labels, center_y):
        if label.ndim != 2:
            raise ValueError(
                f"Expected 2-D label array, got shape {label.shape}. "
                "Check that mask_color_mode='grayscale' in the data generator."
            )

        # Same clamped-offset logic as normalize_inputs
        max_offset = PAD_HEIGHT - crop_height
        offset_height = int(np.clip(center - crop_height // 2, 0, max_offset))

        label = image_utils.pad_images(
            label, target_height=PAD_HEIGHT, target_width=crop_width
        )
        label = image_utils.crop_to_bounding_box(
            images=label,
            offset_height=offset_height,
            offset_width=0,
            crop_height=crop_height,
            crop_width=crop_width
        )

        normalized_labels.append(np.expand_dims(label, axis=0))  # (1, H, W)

    return np.concatenate(normalized_labels, axis=0).astype(np.int32)  # (N, H, W)


def convert_prediction_to_height(y_min, y_max, img_height=PAD_HEIGHT):
    """
    Map normalized bounding-box coordinates to an absolute pixel row.

    The TF Object Detection API returns boxes in normalised [0, 1] coordinates
    with the layout [ymin, xmin, ymax, xmax].  This function converts the
    vertical span to an absolute pixel center used for the crop offset.

    Args:
        y_min      (float): Normalised top edge of the bounding box.
        y_max      (float): Normalised bottom edge of the bounding box.
        img_height (int)  : Full image height in pixels (default: PAD_HEIGHT).

    Returns:
        int: Absolute pixel row of the vertical center of the detection.

    Raises:
        ValueError: If y_min >= y_max (degenerate / inverted bounding box).
    """
    if y_min >= y_max:
        raise ValueError(
            f"Degenerate bounding box: y_min ({y_min:.4f}) must be < "
            f"y_max ({y_max:.4f}).  This detection will be skipped."
        )

    top    = int(y_min * img_height)
    bottom = int(y_max * img_height)
    return top + (bottom - top) // 2  # midpoint in pixel space


def load_graph_def(frozen_graph_path):
    """
    Deserialise a TF1-style frozen inference graph (.pb) from disk.

    This is needed because the MobileNet SSD detector was exported as a
    frozen graph.  In TF2 we still need `tf.compat.v1.GraphDef` for this;
    using plain `tf.GraphDef` raises AttributeError.

    FIX: replaced `tf.GraphDef()` with `tf.compat.v1.GraphDef()`.

    Args:
        frozen_graph_path (str): Filesystem path to the .pb file.

    Returns:
        tf.compat.v1.GraphDef: Parsed protocol-buffer graph definition.

    Raises:
        FileNotFoundError: If the .pb file does not exist at the given path.
        RuntimeError     : If the file cannot be parsed (corrupt / wrong format).
    """
    try:
        with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
            # FIX: bare `tf.GraphDef` does not exist in TF2; must use compat.v1
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Frozen graph not found: {frozen_graph_path}\n"
            f"Absolute path checked: {os.path.abspath(frozen_graph_path)}\n"
            "Download the MobileNet SSD frozen graph and place it there."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse frozen graph at '{frozen_graph_path}': {e}\n"
            "The file may be truncated, corrupted, or in an incompatible format."
        )


# ==============================================================================
# DATA GENERATOR
# ==============================================================================

def model_data_generator(
    parent_dir,
    image_dir,
    mask_dir,
    mobileNet_graphdef,
    load_size,
    batch_size=2,
    image_color_mode="rgb",
    mask_color_mode="grayscale"
):
    """
    Infinite generator that yields (image_batch, mask_batch) pairs ready for
    U-Net training.

    Each iteration:
      1. Load a mini-batch of raw images + masks from disk via
         ImageDataGenerator.
      2. Run MobileNet SSD inference on the raw images to detect the blood
         tube and extract the vertical center of the detected bounding box.
      3. Crop + normalise images  → float32 (N, 368, 368, 3)
      4. Crop + normalise masks   → int32   (N, 368, 368)
      5. yield (images, masks) for one gradient update step.

    FIX — mask interpolation:
        `flow_from_directory` for masks now uses `interpolation='nearest'` to
        preserve discrete pixel values (85, 170, 255) during resizing.
        Bilinear interpolation (the default) blurs label boundaries and
        creates intermediate values that do not map to any class index.

    FIX — bounding-box index iteration:
        `detection_boxes` from sess.run() has shape (batch, N_dets, 4).
        Iterating directly over `boxes_out` yields one array of shape
        (N_dets, 4) per image — not one box per image.  We now take only
        the highest-confidence detection (index 0) per image.

    FIX — per-batch exception isolation:
        A try/except block wraps each individual batch so that a single
        corrupt image or invalid detection does not terminate the generator
        permanently.

    Args:
        parent_dir         (str)             : Root directory that contains
                                               `image_dir` and `mask_dir` as
                                               immediate sub-folders.
        image_dir          (str)             : Sub-folder name for RGB images.
        mask_dir           (str)             : Sub-folder name for mask images.
        mobileNet_graphdef (tf.compat.v1.GraphDef): Pre-loaded frozen graph.
        load_size          (tuple)           : (height, width) to resize images
                                               before MobileNet inference.
        batch_size         (int)             : Mini-batch size.  Default 2.
        image_color_mode   (str)             : 'rgb' or 'grayscale'.
        mask_color_mode    (str)             : Should always be 'grayscale' for
                                               single-channel label maps.

    Yields:
        tuple[np.ndarray, np.ndarray]:
            images : float32 (batch_size, CROP_HEIGHT, CROP_WIDTH, 3)
            masks  : int32   (batch_size, CROP_HEIGHT, CROP_WIDTH)
    """

    # ── Image generator ────────────────────────────────────────────────────
    # No augmentation here; augmentation should go in a separate wrapper if
    # needed, applied only to images (never to masks).
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        parent_dir,
        classes=[image_dir],
        color_mode=image_color_mode,
        target_size=load_size,
        batch_size=batch_size,
        seed=42,          # same seed as mask generator ensures paired shuffling
        shuffle=True,
        class_mode=None
    )

    # ── Mask generator ─────────────────────────────────────────────────────
    # FIX: interpolation='nearest' keeps label pixel values intact.
    # Bilinear (default) would create non-class values like 127 or 212.
    mask_datagen = ImageDataGenerator()
    mask_generator = mask_datagen.flow_from_directory(
        parent_dir,
        classes=[mask_dir],
        color_mode=mask_color_mode,
        target_size=load_size,
        batch_size=batch_size,
        seed=42,           # must match image_generator seed for correct pairing
        shuffle=True,
        class_mode=None,
        interpolation='nearest'   # FIX: preserve discrete label values
    )

    # ── MobileNet SSD session ──────────────────────────────────────────────
    # We open the TF1-compat session once for the lifetime of this generator.
    # The `with` context manager ensures the session is released when the
    # generator object is garbage-collected (GeneratorExit propagates into
    # the `with` block and triggers __exit__).
    with tf.Graph().as_default() as mobilenet_graph:
        with tf.compat.v1.Session(graph=mobilenet_graph) as sess:

            tf.compat.v1.import_graph_def(mobileNet_graphdef, name="mobilenet")

            # Resolve tensor handles once — avoids repeated string lookups
            image_tensor     = mobilenet_graph.get_tensor_by_name('mobilenet/image_tensor:0')
            detection_boxes  = mobilenet_graph.get_tensor_by_name('mobilenet/detection_boxes:0')
            detection_scores = mobilenet_graph.get_tensor_by_name('mobilenet/detection_scores:0')
            detection_classes= mobilenet_graph.get_tensor_by_name('mobilenet/detection_classes:0')
            num_detections   = mobilenet_graph.get_tensor_by_name('mobilenet/num_detections:0')

            for img_batch, mask_batch in zip(image_generator, mask_generator):
                # Per-batch try/except: a corrupt sample or failed detection
                # logs a warning and skips this batch instead of killing the
                # entire generator.
                try:
                    # Run MobileNet inference on the raw (unscaled) batch.
                    # img_batch shape: (batch_size, H, W, 3), float32 [0, 255]
                    boxes_out, scores_out, _, _ = sess.run(
                        [detection_boxes, detection_scores,
                         detection_classes, num_detections],
                        feed_dict={image_tensor: img_batch}
                    )
                    # boxes_out shape: (batch_size, max_detections, 4)
                    # Each row: [ymin, xmin, ymax, xmax] in normalised [0, 1]

                    transition_regions = []

                    # FIX: iterate over images in the batch, not over all
                    # detections.  boxes_out[i] is the (N_dets, 4) array for
                    # image i; we take index [0] = highest-confidence detection.
                    for i in range(len(img_batch)):
                        best_box = boxes_out[i][0]   # shape (4,) — top detection
                        y_min = best_box[0]           # normalised top edge
                        y_max = best_box[2]           # normalised bottom edge
                        # NOTE: box layout is [ymin, xmin, ymax, xmax];
                        #       best_box[1] = xmin and best_box[3] = xmax are
                        #       not used because we only need the vertical center.

                        try:
                            center_y = convert_prediction_to_height(y_min, y_max)
                        except ValueError as e:
                            print(f"  Warning: {e}  → using image midpoint as fallback.")
                            center_y = PAD_HEIGHT // 2

                        transition_regions.append(center_y)

                    # Crop + normalise
                    normalized_images = normalize_inputs(
                        img_batch, center_y=transition_regions
                    )
                    normalized_masks = normalize_labels(
                        mask_batch, center_y=transition_regions
                    )

                    yield (normalized_images, normalized_masks)

                except Exception as e:
                    print(f"  Warning: Skipping batch due to error — {e}")
                    continue  # skip this batch; training proceeds with the next


# ==============================================================================
# TRAINING ORCHESTRATION
# ==============================================================================

def train_unet_model(train_dir, test_dir, num_epochs):
    """
    End-to-end U-Net training pipeline.

    Steps:
      1. Count training and validation images.
      2. Load the frozen MobileNet SSD detector graph.
      3. Construct paired image/mask data generators.
      4. Build and summarise the U-Net model.
      5. Train with ModelCheckpoint callbacks.
      6. Save final model in both SavedModel and legacy H5 formats.

    FIX — ModelCheckpoint path:
        `save_weights_only=True` in TF 2.x requires the filepath to end with
        `.weights.h5`.  The original `.ckpt` extension triggers a deprecation
        warning and may raise an error in Keras 3+.

    Args:
        train_dir  (str): Directory with `Images/` and `Masks/` sub-folders
                          (training split).
        test_dir   (str): Directory with `Images/` and `Masks/` sub-folders
                          (validation split).
        num_epochs (int): Number of full passes over the training set.
    """

    print("=" * 80)
    print("STARTING U-NET TRAINING PIPELINE")
    print("=" * 80)

    # ── 1 · Count dataset sizes ────────────────────────────────────────────
    print("\n[1/6] Counting dataset images...")

    train_image_count = file_io.count_num_files(
        target_dir=os.path.join(train_dir, "Images"),
        file_extension='png'
    )
    test_image_count = file_io.count_num_files(
        target_dir=os.path.join(test_dir, "Images"),
        file_extension='png'
    )

    print(f"  Training images  : {train_image_count}")
    print(f"  Validation images: {test_image_count}")

    # ── 2 · Load MobileNet detector ───────────────────────────────────────
    print("\n[2/6] Loading MobileNet detector graph...")

    mobilenet_graph_path = os.path.join('Graphs', 'mobilenet_frozen_inference_graph.pb')

    if not os.path.exists(mobilenet_graph_path):
        raise FileNotFoundError(
            f"MobileNet graph not found: {mobilenet_graph_path}\n"
            f"Expected absolute path: {os.path.abspath(mobilenet_graph_path)}\n"
            "Please place the frozen-inference-graph .pb file there."
        )

    mobileNet_graphdef = load_graph_def(mobilenet_graph_path)
    print("  ✓ MobileNet loaded")

    # ── 3 · Create data generators ────────────────────────────────────────
    print("\n[3/6] Creating data generators...")

    # FIX: width corrected from 366 → CROP_WIDTH (368) to match the crop
    # target; using 366 caused pad_images to add 2 spurious columns every batch.
    img_dims = (PAD_HEIGHT, CROP_WIDTH)  # (height, width) for flow_from_directory

    train_generator = model_data_generator(
        train_dir, "Images", "Masks",
        load_size=img_dims,
        mobileNet_graphdef=mobileNet_graphdef,
        batch_size=2
    )

    val_generator = model_data_generator(
        test_dir, "Images", "Masks",
        load_size=img_dims,
        mobileNet_graphdef=mobileNet_graphdef,
        batch_size=2
    )

    print("  ✓ Data generators created (batch_size=2)")

    # ── 4 · Build U-Net ───────────────────────────────────────────────────
    print("\n[4/6] Building U-Net model...")

    model = build_unet_model()
    model.summary()

    # ── 5 · Train ─────────────────────────────────────────────────────────
    print("\n[5/6] Training...")

    batch_size           = 2
    train_steps_per_epoch = math.ceil(train_image_count / batch_size)
    val_steps_per_epoch   = math.ceil(test_image_count  / batch_size)

    print(f"  Steps/epoch (train) : {train_steps_per_epoch}")
    print(f"  Steps/epoch (val)   : {val_steps_per_epoch}")
    print(f"  Epochs              : {num_epochs}")

    os.makedirs("UNet_Outputs", exist_ok=True)

    # FIX: `.weights.h5` extension required when save_weights_only=True in
    # TF 2.x / Keras 3+.  The previous `.ckpt` suffix triggered a warning
    # ("ckpt is deprecated") and fails in Keras 3.
    checkpoint_callback = ModelCheckpoint(
        filepath="UNet_Outputs/cp-{epoch:04d}.weights.h5",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch',
        monitor='val_loss',
        mode='min'
    )

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=num_epochs,
        callbacks=[checkpoint_callback],
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps_per_epoch,
        verbose=1
    )

    print("\n  ✓ Training complete!")

    # ── 6 · Save model ────────────────────────────────────────────────────
    print("\n[6/6] Saving model...")

    # SavedModel format — recommended for TF 2.x serving and further training
    saved_model_path = "UNet_Outputs/unet_model"
    model.save(saved_model_path)
    print(f"  ✓ SavedModel → {saved_model_path}/")

    # H5 format — legacy; useful for quick weight transfers with model.load_weights()
    model.save("UNet_Outputs/unet.h5")
    print("  ✓ H5 model  → UNet_Outputs/unet.h5")

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput artefacts:")
    print("  UNet_Outputs/unet_model/           — SavedModel (recommended)")
    print("  UNet_Outputs/unet.h5               — Keras H5 (legacy)")
    print("  UNet_Outputs/cp-NNNN.weights.h5    — Best checkpoint per epoch")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

def main():
    """Parse command-line arguments, validate them, and launch training."""

    parser = argparse.ArgumentParser(
        description='Train a U-Net segmentation model for blood layer analysis.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unet_training_consolidated.py \\
      --train_dir data/train --test_dir data/val --epochs 100

  python unet_training_consolidated.py \\
      --train_dir /mnt/ssd/train --test_dir /mnt/ssd/val --epochs 50

Directory structure expected:
  <train_dir>/
      Images/   ← RGB PNG images of centrifuge tubes
      Masks/    ← Grayscale PNG masks (0 / 85 / 170 / 255)
  <test_dir>/
      Images/
      Masks/
        """
    )

    parser.add_argument(
        '--train_dir', required=True, type=str,
        help='Training directory containing Images/ and Masks/ sub-folders.'
    )
    parser.add_argument(
        '--test_dir', required=True, type=str,
        help='Validation directory containing Images/ and Masks/ sub-folders.'
    )
    parser.add_argument(
        '--epochs', required=True, type=int,
        help='Number of training epochs (positive integer).'
    )

    args = parser.parse_args()

    # Validate paths and epoch count before doing any heavy work
    if not os.path.isdir(args.train_dir):
        raise ValueError(f"Training directory not found: {args.train_dir}")
    if not os.path.isdir(args.test_dir):
        raise ValueError(f"Validation directory not found: {args.test_dir}")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be a positive integer, got: {args.epochs}")

    train_unet_model(args.train_dir, args.test_dir, args.epochs)


if __name__ == '__main__':
    main()
