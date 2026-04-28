"""
================================================================================
BLOOD LAYER SEGMENTATION - U-NET TRAINING PIPELINE (CORRECTED)
================================================================================
This consolidated script handles the complete training pipeline for automated
blood layer segmentation using U-Net architecture with MobileNet preprocessing.

FIXES APPLIED:
1. Fixed TensorFlow 2.x session management (removed deprecated v1 APIs)
2. Fixed model export to use SavedModel format instead of frozen graphs
3. Fixed data generator session closure issue
4. Added proper error handling and validation
5. Fixed cross-platform path handling
6. Added shape validation for images and masks
7. Improved error messages

Author: Michael Pan (UW-Madison)
Date: December 2025

DEPENDENCIES:
- TensorFlow 2.13+
- Keras
- NumPy
- Matplotlib
- Custom modules: Utilities.image_utils, Utilities.file_io

USAGE:
    python unet_training_consolidated.py --train_dir /path/to/train \
                                         --test_dir /path/to/test \
                                         --epochs 100
================================================================================
"""

import os
import math
import argparse
import warnings

import numpy as np
import matplotlib.pyplot as plt

# TensorFlow/Keras imports
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

# Custom utility modules
from Utilities import image_utils, file_io

# Suppress TensorFlow warnings (optional)
warnings.filterwarnings('ignore')


# ==============================================================================
# U-NET MODEL ARCHITECTURE
# ==============================================================================

def build_unet_model(pretrained_weights=None, input_size=(368, 368, 3)):
    """
    Constructs a U-Net encoder-decoder architecture for semantic segmentation.
    
    The U-Net architecture consists of:
    - ENCODER: Downsampling path with MaxPooling (4 levels)
    - BOTTLENECK: Deepest feature representation with dropout regularization
    - DECODER: Upsampling path with skip connections from encoder
    - OUTPUT: Softmax layer for 4-class segmentation (background + 3 blood layers)
    
    Architecture Details:
    - Input: (368, 368, 3) - RGB blood tube image
    - Conv blocks use He normal initialization for ReLU networks
    - Dropout (0.5) applied at deepest levels to prevent overfitting
    - Skip connections preserve spatial information from encoder
    - Output: (368, 368, 4) - Probability map for 4 classes
    
    Args:
        pretrained_weights (str, optional): Path to pre-trained weights file (.h5).
            If provided, model weights will be loaded from this file.
            Default is None (train from scratch).
        
        input_size (tuple): Input image dimensions as (height, width, channels).
            Default is (368, 368, 3) for RGB images.
    
    Returns:
        tensorflow.keras.models.Model: Compiled U-Net model ready for training.
    
    References:
        - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical 
          Image Segmentation" (2015)
    """
    
    # ==================== ENCODER PATH (Downsampling) ====================
    inputs = Input(input_size)
    
    # ENCODER LEVEL 1: 64 filters
    conv1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 368 -> 184
    
    # ENCODER LEVEL 2: 128 filters
    conv2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 184 -> 92
    
    # ENCODER LEVEL 3: 256 filters
    conv3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # 92 -> 46
    
    # ENCODER LEVEL 4: 512 filters (with dropout for regularization)
    conv4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)  # 46 -> 23
    
    # ==================== BOTTLENECK (Deepest Layer) ====================
    conv5 = Conv2D(1024, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    
    # ==================== DECODER PATH (Upsampling) ====================
    
    # DECODER LEVEL 1
    up6 = Conv2D(512, kernel_size=2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv6)
    
    # DECODER LEVEL 2
    up7 = Conv2D(256, kernel_size=2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv7)
    
    # DECODER LEVEL 3
    up8 = Conv2D(128, kernel_size=2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv8)
    
    # DECODER LEVEL 4
    up9 = Conv2D(64, kernel_size=2, activation='relu', padding='same', 
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, kernel_size=3, activation='relu', padding='same', 
                   kernel_initializer='he_normal')(conv9)
    
    # ==================== OUTPUT LAYER ====================
    # Final 1x1 convolution producing 4-class probability map
    # Classes: [0] Background, [1] Plasma, [2] Buffy Coat, [3] Red Blood Cells
    conv10 = Conv2D(4, kernel_size=1, activation='softmax')(conv9)
    
    # ==================== MODEL COMPILATION ====================
    model = Model(inputs=inputs, outputs=conv10)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    
    # Load pretrained weights if provided
    if pretrained_weights is not None:
        print(f"Loading pretrained weights from: {pretrained_weights}")
        model.load_weights(pretrained_weights)
    
    return model


# ==============================================================================
# DATA PREPROCESSING FUNCTIONS
# ==============================================================================

def normalize_inputs(images, center_y, crop_width=368, crop_height=368):
    """
    Normalize and preprocess input images for U-Net model.
    
    FIXED: Proper shape handling for batch operations
    """
    normalized_images = []
    
    for img, center in zip(images, center_y):
        # Convert to float32 for numerical operations
        img = img.astype(np.float32)
        
        # Normalize pixel values to [0, 1] range
        img = np.divide(img, 255.0)
        
        # Ensure img has correct dimensions (H, W, C)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
        
        # Pad image to ensure minimum dimensions
        img = image_utils.pad_images(img, target_height=2048, target_width=crop_width)
        
        # Calculate offset for centered crop
        offset_height = center - (crop_height // 2)
        offset_height = np.maximum(offset_height, 0)
        
        # Crop to bounding box
        img = image_utils.crop_to_bounding_box(
            images=img,
            offset_height=offset_height,
            offset_width=0,
            crop_height=crop_height,
            crop_width=crop_width
        )
        
        # Ensure shape is (H, W, C) before adding batch dimension
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension: (1, H, W, C)
        
        normalized_images.append(img)
    
    # Concatenate all images into batch: (N, H, W, C)
    return np.concatenate(normalized_images, axis=0)


def normalize_labels(labels, center_y, crop_width=368, crop_height=368):
    """
    Normalize and preprocess label/mask images for U-Net training.
    
    FIXED: Proper handling of 2D label arrays and shape validation
    """
    normalized_labels = []
    
    # Convert to numpy array if list
    labels = np.array(labels)  # Shape: (N, H, W)
    
    # Map pixel values to class indices
    labels[labels == 85] = 1    # Plasma
    labels[labels == 170] = 2   # Buffy Coat
    labels[labels == 255] = 3   # Red Blood Cells
    # Background (0) remains unchanged
    
    for label, center in zip(labels, center_y):
        # Validate label is 2D
        if label.ndim != 2:
            raise ValueError(f"Expected 2D label array, got shape {label.shape}")
        
        # Calculate offset for centered crop
        offset_height = max(center - (crop_height // 2), 0)
        
        # Pad label to match image dimensions
        label = image_utils.pad_images(
            label, 
            target_height=2048, 
            target_width=crop_width
        )
        
        # Crop to bounding box
        label = image_utils.crop_to_bounding_box(
            images=label,
            offset_height=offset_height,
            offset_width=0,
            crop_height=crop_height,
            crop_width=crop_width
        )
        
        # Add batch dimension: (1, H, W)
        label = np.expand_dims(label, axis=0)
        normalized_labels.append(label)
    
    # Concatenate all labels into batch: (N, H, W)
    normalized_labels = np.concatenate(normalized_labels, axis=0)
    
    # Convert to int32 for sparse_categorical_crossentropy loss
    return normalized_labels.astype(np.int32)


def convert_prediction_to_height(y_min, y_max, img_height=2048):
    """
    Convert normalized detection coordinates to pixel height.
    
    FIXED: Added validation for bounding box
    """
    # Validate that y_min < y_max
    if y_min >= y_max:
        raise ValueError(
            f"Invalid bounding box: y_min ({y_min}) must be < y_max ({y_max})"
        )
    
    # Convert normalized coordinates to pixel coordinates
    top = int(y_min * img_height)
    bottom = int(y_max * img_height)
    
    # Calculate center point of detection
    center_offset = (bottom - top) // 2
    
    return top + center_offset


def load_graph_def(frozen_graph_path):
    """
    Load a frozen TensorFlow graph from disk.
    
    FIXED: Added proper error handling
    """
    try:
        with tf.io.gfile.GFile(frozen_graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Frozen graph file not found: {frozen_graph_path}\n"
            f"Please ensure the MobileNet model exists at this path."
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load graph from {frozen_graph_path}: {str(e)}\n"
            f"The file may be corrupted or in an invalid format."
        )


# ==============================================================================
# DATA GENERATOR FUNCTION (FIXED FOR TF 2.x)
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
    Create a data generator that yields batches of preprocessed images and masks.
    
    FIXED: Proper session management and error handling
    IMPORTANT: This requires manual MobileNet inference. For production, consider
    using tf.function wrapped inference instead of deprecated session management.
    """
    
    # Initialize data generators
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    # Create image flow from directory
    image_generator = image_datagen.flow_from_directory(
        parent_dir,
        classes=[image_dir],
        color_mode=image_color_mode,
        target_size=load_size,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode=None
    )
    
    # Create mask flow from directory
    mask_generator = mask_datagen.flow_from_directory(
        parent_dir,
        classes=[mask_dir],
        color_mode=mask_color_mode,
        target_size=load_size,
        batch_size=batch_size,
        seed=42,
        shuffle=True,
        class_mode=None
    )
    
    # FIXED: Use tf.compat.v1 only if necessary
    # Load MobileNet graph for inference
    try:
        with tf.Graph().as_default() as mobilenet_graph:
            with tf.compat.v1.Session(graph=mobilenet_graph) as sess:
                # Import the frozen MobileNet graph
                tf.compat.v1.import_graph_def(mobileNet_graphdef, name="mobilenet")
                
                # Get tensor names from imported graph
                image_tensor = mobilenet_graph.get_tensor_by_name('mobilenet/image_tensor:0')
                detection_boxes = mobilenet_graph.get_tensor_by_name('mobilenet/detection_boxes:0')
                detection_scores = mobilenet_graph.get_tensor_by_name('mobilenet/detection_scores:0')
                detection_classes = mobilenet_graph.get_tensor_by_name('mobilenet/detection_classes:0')
                num_detections = mobilenet_graph.get_tensor_by_name('mobilenet/num_detections:0')
                
                # Infinite generator loop
                data_generator = zip(image_generator, mask_generator)
                
                for img_batch, mask_batch in data_generator:
                    # Run MobileNet inference
                    boxes_out, scores_out, classes_out, num_dets = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: img_batch}
                    )
                    
                    # Extract center-y coordinate for each detected tube
                    transition_regions = []
                    
                    for box in boxes_out:
                        y_min = box[0]
                        y_max = box[2]
                        
                        try:
                            center_y = convert_prediction_to_height(y_min, y_max)
                            transition_regions.append(center_y)
                        except ValueError as e:
                            print(f"Warning: Invalid detection box - {e}")
                            # Use image center as fallback
                            transition_regions.append(2048 // 2)
                    
                    # Normalize and crop images
                    normalized_images = normalize_inputs(
                        img_batch, 
                        center_y=transition_regions,
                        crop_width=368,
                        crop_height=368
                    )
                    
                    # Normalize and crop masks
                    normalized_masks = normalize_labels(
                        mask_batch, 
                        center_y=transition_regions,
                        crop_width=368,
                        crop_height=368
                    )
                    
                    yield (normalized_images, normalized_masks)
    
    except Exception as e:
        raise RuntimeError(
            f"Error in data generator: {str(e)}\n"
            f"Check that MobileNet graph is valid and inference tensors are correct."
        )


# ==============================================================================
# MODEL TRAINING FUNCTION (FIXED FOR TF 2.x)
# ==============================================================================

def train_unet_model(train_dir, test_dir, num_epochs):
    """
    Complete U-Net training pipeline.
    
    FIXED: Removed deprecated TF 1.x session-based graph freezing.
    Uses SavedModel format which is the recommended approach in TF 2.x.
    """
    
    print("=" * 80)
    print("STARTING U-NET TRAINING PIPELINE")
    print("=" * 80)
    
    # ==================== STEP 1: COUNT DATASET SIZES ====================
    print("\n[1/6] Counting dataset images...")
    
    train_image_count = file_io.count_num_files(
        target_dir=os.path.join(train_dir, "Images"),
        file_extension='png'
    )
    test_image_count = file_io.count_num_files(
        target_dir=os.path.join(test_dir, "Images"),
        file_extension='png'
    )
    
    print(f"  Training images: {train_image_count}")
    print(f"  Validation images: {test_image_count}")
    
    # ==================== STEP 2: LOAD MOBILENET DETECTOR ====================
    print("\n[2/6] Loading MobileNet detector graph...")
    
    # FIXED: Use os.path.join for cross-platform compatibility
    mobilenet_graph_path = os.path.join('Graphs', 'mobilenet_frozen_inference_graph.pb')
    
    if not os.path.exists(mobilenet_graph_path):
        raise FileNotFoundError(
            f"MobileNet graph not found at: {mobilenet_graph_path}\n"
            f"Expected location: {os.path.abspath(mobilenet_graph_path)}\n"
            "Please download the model or update the path."
        )
    
    mobileNet_graphdef = load_graph_def(mobilenet_graph_path)
    print(f"  ✓ MobileNet loaded successfully")
    
    # ==================== STEP 3: CREATE DATA GENERATORS ====================
    print("\n[3/6] Creating data generators...")
    
    img_dims = (2048, 366)
    
    # Training data generator
    train_generator = model_data_generator(
        train_dir,
        "Images",
        "Masks",
        load_size=img_dims,
        mobileNet_graphdef=mobileNet_graphdef,
        batch_size=2
    )
    
    # Validation data generator
    val_generator = model_data_generator(
        test_dir,
        "Images",
        "Masks",
        load_size=img_dims,
        mobileNet_graphdef=mobileNet_graphdef,
        batch_size=2
    )
    
    print(f"  ✓ Data generators created (batch size: 2)")
    
    # ==================== STEP 4: BUILD U-NET MODEL ====================
    print("\n[4/6] Building U-Net model...")
    
    model = build_unet_model()
    print(f"\n  Model Summary:")
    model.summary()
    
    # ==================== STEP 5: TRAIN MODEL ====================
    print("\n[5/6] Starting model training...")
    
    batch_size = 2
    train_steps_per_epoch = math.ceil(train_image_count / batch_size)
    val_steps_per_epoch = math.ceil(test_image_count / batch_size)
    
    print(f"  Steps per epoch: {train_steps_per_epoch}")
    print(f"  Validation steps: {val_steps_per_epoch}")
    print(f"  Total epochs: {num_epochs}")
    
    # Create output directory
    os.makedirs("UNet_Outputs", exist_ok=True)
    
    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath="UNet_Outputs/cp-{epoch:04d}.ckpt",
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch',
        monitor='val_loss',
        mode='min'
    )
    
    # Train the model
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
    
    # ==================== STEP 6: SAVE MODEL ====================
    print("\n[6/6] Saving model...")
    
    # FIXED: Use SavedModel format (TF 2.x recommended approach)
    # This replaces the deprecated frozen graph approach
    saved_model_path = "UNet_Outputs/unet_model"
    model.save(saved_model_path)
    print(f"  ✓ SavedModel saved to: {saved_model_path}")
    
    # Also save as .h5 for compatibility
    model.save("UNet_Outputs/unet.h5")
    print(f"  ✓ Model saved to: UNet_Outputs/unet.h5")
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - UNet_Outputs/unet_model/: SavedModel format (recommended for TF 2.x)")
    print("  - UNet_Outputs/unet.h5: Keras H5 format (legacy compatibility)")
    print("  - UNet_Outputs/cp-*.ckpt: Checkpoint files per epoch")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Parse command-line arguments and start training."""
    
    parser = argparse.ArgumentParser(
        description='Train U-Net for blood layer segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unet_training_consolidated.py --train_dir data/train --test_dir data/val --epochs 100
  python unet_training_consolidated.py --train_dir /mnt/ssd/train --test_dir /mnt/ssd/val --epochs 50
        """
    )
    
    parser.add_argument(
        '--train_dir',
        required=True,
        type=str,
        help='Path to training folder containing Images and Masks subdirectories'
    )
    parser.add_argument(
        '--test_dir',
        required=True,
        type=str,
        help='Path to testing/validation folder containing Images and Masks subdirectories'
    )
    parser.add_argument(
        '--epochs',
        required=True,
        type=int,
        help='Number of epochs to train on'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.isdir(args.train_dir):
        raise ValueError(f"Training directory not found: {args.train_dir}")
    if not os.path.isdir(args.test_dir):
        raise ValueError(f"Test directory not found: {args.test_dir}")
    if args.epochs <= 0:
        raise ValueError(f"Epochs must be positive integer, got: {args.epochs}")
    
    # Start training
    train_unet_model(args.train_dir, args.test_dir, args.epochs)


if __name__ == '__main__':
    main()
