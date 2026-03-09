"""
Activation map generation with preprocessing options.

Generates Guided Grad-CAM visualizations for CNN model predictions.
Supports both image directories and TFRecord inputs.

Usage:
    # From image directory
    python gradcam_refactored.py --im_dir /path/to/images --model_path model.keras --resdir /output

    # From TFRecord
    python gradcam_refactored.py --deploy_tfrec test.tfrecord --model_path model.keras --resdir /output
"""

import os
import sys
import glob
import argparse
import logging
from pathlib import Path
from typing import Generator, Tuple, List, Optional, Dict, Any

import numpy as np
import pandas as pd
import pyfiglet
import tensorflow as tf
from imageio import imread, imwrite
from PIL import Image

from activationmap.grads import Grads
from activationmap.grad_ops import GradOps
import param_gedi as param
from preprocessing.datagenerator import Dataspring

__version__ = '2.0.0'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Architecture-specific layer presets
LAYER_PRESETS = {
    'resnet50': 'conv5_block3_out',
    'resnet101': 'conv5_block3_out',
    'vgg16': 'block5_conv3',
    'vgg19': 'block5_conv4',
    'custom': 'conv2d_4',
}


class GradCAMConfig:
    """Configuration container for Grad-CAM processing."""
    
    def __init__(self,
                 model_path: str,
                 output_dir: str,
                 image_dir: Optional[str] = None,
                 tfrecord_path: Optional[str] = None,
                 layer_name: str = 'conv5_block3_out',
                 batch_size: int = 32,
                 image_type: str = 'tif',
                 guided: bool = True,
                 vgg_normalize: bool = True,
                 has_labels: bool = True,
                 expected_label: int = 1,
                 heatmap_only: bool = False,
                 save_cropped: bool = True,
                 target_size: Tuple[int, int] = (224, 224),
                 orig_size: Tuple[int, int] = (512, 512)):
        """
        Initialize Grad-CAM configuration.
        
        Args:
            model_path: Path to trained Keras model
            output_dir: Directory for output visualizations
            image_dir: Directory containing images (mutually exclusive with tfrecord_path)
            tfrecord_path: Path to TFRecord file (mutually exclusive with image_dir)
            layer_name: Target layer for Grad-CAM visualization
            batch_size: Number of images per batch
            image_type: Image file extension (tif, jpg, png)
            guided: Whether to use Guided Grad-CAM
            vgg_normalize: Whether to apply VGG normalization
            has_labels: Whether images have associated labels
            save_cropped: Whether to save cropped versions of original images
            target_size: Target size after cropping (height, width)
            orig_size: Original size before cropping (height, width)
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.image_dir = image_dir
        self.tfrecord_path = tfrecord_path
        self.layer_name = layer_name
        self.batch_size = batch_size
        self.image_type = image_type
        self.guided = guided
        self.vgg_normalize = vgg_normalize
        self.has_labels = has_labels
        self.expected_label = expected_label
        self.heatmap_only = heatmap_only
        self.save_cropped = save_cropped
        self.target_size = target_size
        self.orig_size = orig_size

        # Confusion matrix directory structure
        self.conf_mat_paths = [
            ['zero_true', 'zero_false'],
            ['one_false', 'one_true']
        ]
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if (self.image_dir is None) == (self.tfrecord_path is None):
            raise ValueError("Exactly one of image_dir or tfrecord_path must be specified")
        
        if self.image_dir and not os.path.exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        if self.tfrecord_path and not os.path.exists(self.tfrecord_path):
            raise FileNotFoundError(f"TFRecord not found: {self.tfrecord_path}")
        
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")


class ImageBatchGenerator:
    """Generates batches of images with optional labels."""
    
    def __init__(self, config: GradCAMConfig, parser=None):
        """
        Initialize batch generator.
        
        Args:
            config: GradCAMConfig object
            parser: Optional image preprocessing function
        """
        self.config = config
        self.parser = parser or (lambda x: x)
    
    def from_directory(self,
                       source_dir: str,
                       dead_dir: Optional[str] = None,
                       live_dir: Optional[str] = None) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate batches from image directory.

        Args:
            source_dir: Directory containing images
            dead_dir: Directory with dead cell labels (optional)
            live_dir: Directory with live cell labels (optional)

        Yields:
            Tuple of (processed_images, labels, filenames, raw_images)
        """
        pattern = os.path.join(source_dir, f'*.{self.config.image_type}')
        image_files = glob.glob(pattern)
        
        if not image_files:
            logger.warning(f"No {self.config.image_type} files found in {source_dir}")
            return
        
        logger.info(f"Found {len(image_files)} images in {source_dir}")
        
        files, labels = [], []
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            
            if self.config.has_labels and dead_dir and live_dir:
                label = self._get_label(filename, dead_dir, live_dir)
                if label is None:
                    continue
            else:
                # Use expected label for unlabeled data
                if self.config.expected_label == 0:
                    label = [1, 0]  # Label 0: [dead=1, live=0]
                else:
                    label = [0, 1]  # Label 1: [dead=0, live=1]
            
            files.append(image_path)
            labels.append(label)
        
        # Generate batches
        for i in range(0, len(files), self.config.batch_size):
            batch_files = files[i:i + self.config.batch_size]
            batch_labels = labels[i:i + self.config.batch_size]
            
            try:
                batch_images, batch_raw_images = self._load_images(batch_files)
                batch_names = [Path(f).stem for f in batch_files]

                yield (
                    np.array(batch_images),
                    np.array(batch_labels),
                    np.array(batch_names),
                    np.array(batch_raw_images)
                )
            except Exception as e:
                logger.error(f"Error loading batch starting at index {i}: {e}")
                continue
    
    def from_tfrecord(self, param_obj, tfrecord_path: str) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate batches from TFRecord.

        Args:
            param_obj: Parameter object for Dataspring
            tfrecord_path: Path to TFRecord file

        Yields:
            Tuple of (images, labels, filenames, raw_images)
        """
        try:
            dataspring = Dataspring(param_obj, tfrecord_path)
            dataset = dataspring.datagen_base(istraining=False, count=1)
            
            for imgs, lbls, files in dataset:
                imgs_np = imgs.numpy()
                lbls_np = lbls.numpy()
                files_np = files.numpy()
                
                # Decode filenames
                filenames = [f.decode() for f in files_np]
                names = np.array([Path(f).stem for f in filenames])

                # For TFRecord, imgs_np is already raw data before processing
                yield (imgs_np, lbls_np, names, imgs_np.copy())
                
                # Explicit cleanup
                del imgs_np, lbls_np, files_np
                
        except Exception as e:
            logger.error(f"Error reading TFRecord {tfrecord_path}: {e}")
            raise
    
    def _get_label(self, filename: str, dead_dir: str, live_dir: str) -> Optional[List[int]]:
        """
        Determine label for an image based on presence in label directories.
        
        Args:
            filename: Image filename
            dead_dir: Directory containing dead cell labels
            live_dir: Directory containing live cell labels
            
        Returns:
            One-hot encoded label [dead, live] or None if ambiguous
        """
        is_dead = os.path.exists(os.path.join(dead_dir, filename))
        is_live = os.path.exists(os.path.join(live_dir, filename))
        
        # Image must have exactly one label
        if is_dead == is_live:
            logger.warning(f"Ambiguous label for {filename}: dead={is_dead}, live={is_live}")
            return None
        
        return [int(is_dead), int(is_live)]
    
    def _load_images(self, file_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load and preprocess a list of images.

        Args:
            file_paths: List of image file paths

        Returns:
            Tuple of (processed_images, raw_images)
        """
        processed_images = []
        raw_images = []
        for path in file_paths:
            try:
                # Load raw image preserving original bit depth
                if path.lower().endswith('.tif') or path.lower().endswith('.tiff'):
                    # Use PIL for TIFF files to preserve bit depth
                    pil_img = Image.open(path)
                    raw_img = np.array(pil_img)
                    pil_img.close()
                else:
                    # Use imageio for other formats
                    raw_img = imread(path)

                processed_img = self.parser(raw_img)
                raw_images.append(raw_img)
                processed_images.append(processed_img)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                raise
        return processed_images, raw_images


class GradCAMProcessor:
    """Processes images and generates Grad-CAM visualizations."""
    
    def __init__(self, config: GradCAMConfig):
        """
        Initialize Grad-CAM processor.
        
        Args:
            config: GradCAMConfig object
        """
        self.config = config
        self.grads = Grads(config.model_path, guidedbool=config.guided, heatmap_only=config.heatmap_only)
        self.grad_ops = GradOps(vgg_normalize=config.vgg_normalize)
        self.batch_generator = ImageBatchGenerator(config, parser=self.grad_ops.img_parse)
        
        logger.info(f"Initialized Grad-CAM with model: {config.model_path}")
        logger.info(f"Target layer: {config.layer_name}")
        logger.info(f"Guided mode: {config.guided}")
    
    def _crop_image(self, image: np.ndarray) -> np.ndarray:
        """
        Crop image from original size to target size using center cropping.

        Args:
            image: Original image array

        Returns:
            Cropped image array
        """
        if len(image.shape) == 2:
            # Grayscale image
            h, w = image.shape
        elif len(image.shape) == 3:
            # Color image or single channel with explicit dimension
            h, w = image.shape[:2]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        target_h, target_w = self.config.target_size
        orig_h, orig_w = self.config.orig_size

        # Only crop if original size is larger than target size
        if h > target_h or w > target_w:
            # Calculate center crop coordinates
            y0 = (h - target_h) // 2 if h > target_h else 0
            x0 = (w - target_w) // 2 if w > target_w else 0

            # Ensure we don't exceed image bounds
            y1 = min(y0 + target_h, h)
            x1 = min(x0 + target_w, w)
            y0 = max(0, y1 - target_h)
            x0 = max(0, x1 - target_w)

            if len(image.shape) == 2:
                cropped = image[y0:y1, x0:x1]
            else:
                cropped = image[y0:y1, x0:x1, :]

            # Save raw cropped image here before any processing
            if hasattr(self, '_current_filename') and hasattr(self, '_current_output_dir'):
                try:
                    from pathlib import Path
                    cropped_filename = f"{self._current_filename}_raw_cropped.tif"
                    cropped_output_path = Path(self._current_output_dir) / cropped_filename

                    # Convert to appropriate integer type for PIL if needed
                    if cropped.dtype == np.float32 or cropped.dtype == np.float64:
                        if cropped.max() <= 1.0:
                            cropped_int = (cropped * 65535).astype(np.uint16)
                        else:
                            cropped_int = cropped.astype(np.uint16)
                    else:
                        cropped_int = cropped

                    # Remove singleton dimensions and convert to 2D for grayscale
                    if len(cropped_int.shape) > 2:
                        if cropped_int.shape[-1] == 1:
                            cropped_int = cropped_int.squeeze(-1)
                        elif cropped_int.shape[-1] == 3 and np.allclose(cropped_int[..., 0], cropped_int[..., 1]) and np.allclose(cropped_int[..., 1], cropped_int[..., 2]):
                            cropped_int = cropped_int[..., 0]

                    # Save using PIL for TIFF
                    pil_img = Image.fromarray(cropped_int)
                    pil_img.save(str(cropped_output_path))
                    pil_img.close()
                    logger.debug(f"Saved raw cropped image before processing: {cropped_output_path}")
                except Exception as e:
                    logger.warning(f"Failed to save raw cropped image: {e}")

            return cropped

        return image


    def process_batch(self,
                      images: np.ndarray,
                      labels: np.ndarray,
                      filenames: np.ndarray,
                      raw_images: np.ndarray) -> Dict[str, List]:
        """
        Process a batch of images and save Grad-CAM visualizations.
        Now processes images one by one instead of as a batch.

        Args:
            images: Batch of processed images
            labels: Batch of labels (one-hot encoded)
            filenames: Batch of filenames
            raw_images: Batch of raw images before processing

        Returns:
            Dictionary with processing results
        """
        results = {'filename': [], 'label': [], 'prediction': [], 'saved': []}

        # Process each image individually instead of as a batch
        for i, (image, label, filename, raw_image) in enumerate(zip(images, labels, filenames, raw_images)):
            try:
                # Set context for _crop_image method to save raw cropped images
                self._current_filename = filename
                self._current_output_dir = self.config.output_dir

                # Process single image (expand dims to make it a batch of 1)
                single_image = np.expand_dims(image, axis=0)
                single_label = np.expand_dims(label, axis=0)

                # Generate Grad-CAM for single image
                ggcam_gen = self.grads.gen_ggcam_stacks(
                    single_image,
                    single_label,
                    self.config.layer_name,
                    ret_preds=True
                )

                # Get the single result
                grad_stack, predictions = next(ggcam_gen)

                success = self._save_visualization(
                    grad_stack,
                    label,
                    predictions,
                    filename,
                    raw_image=raw_image if self.config.save_cropped else None,
                    processed_image=image if self.config.save_cropped and not self.config.heatmap_only else None
                )

                results['filename'].append(filename)
                results['label'].append(np.argmax(label))
                results['prediction'].append(np.argmax(predictions))
                results['saved'].append(success)

            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                results['filename'].append(filename)
                results['label'].append(np.argmax(label))
                results['prediction'].append(-1)  # Only set -1 on actual errors
                results['saved'].append(False)

        return results
    
    def _save_visualization(self,
                           grad_stack: np.ndarray,
                           label: np.ndarray,
                           predictions: np.ndarray,
                           filename: str,
                           raw_image: Optional[np.ndarray] = None,
                           processed_image: Optional[np.ndarray] = None) -> bool:
        """
        Save Grad-CAM visualization to appropriate directory.

        Args:
            grad_stack: Grad-CAM visualization array
            label: True label (one-hot)
            predictions: Model predictions
            filename: Output filename
            raw_image: Raw image before processing (optional)
            processed_image: Processed image for overlay mode (optional)

        Returns:
            True if save successful, False otherwise
        """
        true_label = np.argmax(label)
        pred_label = np.argmax(predictions)
        
        # Determine output path based on confusion matrix
        is_correct = int(true_label == pred_label)
        subdir = self.config.conf_mat_paths[true_label][is_correct]
        
        output_path = Path(self.config.output_dir) / subdir / f"{filename}.tif"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the Grad-CAM visualization
            imwrite(str(output_path), grad_stack)

            # Save cropped version of raw image if requested (always save raw cropped when save_cropped=True)
            if self.config.save_cropped and raw_image is not None:
                # Create cropped image filename
                cropped_filename = f"{filename}_cropped.tif"
                cropped_output_path = Path(self.config.output_dir) / subdir / cropped_filename

                # Crop the raw image (before any RGB conversion or normalization)
                cropped_raw_image = self._crop_image(raw_image)

                # Save raw cropped image preserving bit depth
                if str(cropped_output_path).lower().endswith('.tif'):
                    # Convert to appropriate integer type for PIL if needed
                    if cropped_raw_image.dtype == np.float32 or cropped_raw_image.dtype == np.float64:
                        # If float type, assume it's normalized 0-1 and convert to 16-bit
                        if cropped_raw_image.max() <= 1.0:
                            cropped_raw_image_int = (cropped_raw_image * 65535).astype(np.uint16)
                        else:
                            # If values > 1, assume raw values and convert to uint16
                            cropped_raw_image_int = cropped_raw_image.astype(np.uint16)
                    else:
                        cropped_raw_image_int = cropped_raw_image

                    # Remove singleton dimensions and convert to 2D for grayscale
                    if len(cropped_raw_image_int.shape) > 2:
                        if cropped_raw_image_int.shape[-1] == 1:
                            cropped_raw_image_int = cropped_raw_image_int.squeeze(-1)
                        elif cropped_raw_image_int.shape[-1] == 3 and np.allclose(cropped_raw_image_int[..., 0], cropped_raw_image_int[..., 1]) and np.allclose(cropped_raw_image_int[..., 1], cropped_raw_image_int[..., 2]):
                            # All channels are the same, convert to grayscale
                            cropped_raw_image_int = cropped_raw_image_int[..., 0]

                    # Use PIL to preserve bit depth for TIFF files
                    pil_img = Image.fromarray(cropped_raw_image_int)
                    pil_img.save(str(cropped_output_path))
                    pil_img.close()
                else:
                    # Use imageio for other formats
                    imwrite(str(cropped_output_path), cropped_raw_image)
                logger.info(f"Saved raw cropped image: {cropped_output_path}")
            else:
                logger.info(f"Not saving cropped image: save_cropped={self.config.save_cropped}, raw_image={'present' if raw_image is not None else 'None'}")

            # Save cropped version of processed image for overlay mode if needed
            if self.config.save_cropped and processed_image is not None and not self.config.heatmap_only:
                # Create processed cropped image filename
                processed_cropped_filename = f"{filename}_processed_cropped.tif"
                processed_cropped_output_path = Path(self.config.output_dir) / subdir / processed_cropped_filename

                # Crop the processed image
                cropped_processed_image = self._crop_image(processed_image)

                # Save processed cropped image
                imwrite(str(processed_cropped_output_path), cropped_processed_image)
                logger.debug(f"Saved processed cropped image: {processed_cropped_output_path}")

            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to save {output_path}: {e}")
            return False
    
    def run(self) -> pd.DataFrame:
        """
        Run Grad-CAM processing on all images.
        
        Returns:
            DataFrame with processing results
        """
        all_results = pd.DataFrame({
            'filename': [],
            'label': [],
            'prediction': [],
            'saved': []
        })
        
        # Determine batch source
        if self.config.image_dir:
            logger.info(f"Processing images from directory: {self.config.image_dir}")
            batch_gen = self._process_from_directory()
        else:
            logger.info(f"Processing images from TFRecord: {self.config.tfrecord_path}")
            batch_gen = self._process_from_tfrecord()
        
        # Process all batches
        batch_count = 0
        for images, labels, names, raw_images in batch_gen:
            batch_count += 1
            logger.info(f"Processing batch {batch_count} ({len(names)} images), starting with: {names[0]}")

            results = self.process_batch(images, labels, names, raw_images)
            batch_df = pd.DataFrame(results)

            # Don't override labels to -1, keep the actual predictions and labels

            all_results = pd.concat([all_results, batch_df], ignore_index=True)

            # Explicit cleanup
            del images, labels, names, raw_images
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _process_from_directory(self) -> Generator:
        """Process images from directory structure."""
        subdirs = glob.glob(os.path.join(self.config.image_dir, '**'))
        
        # Check if directory has subdirectories
        if subdirs and os.path.isdir(subdirs[0]):
            logger.info(f"Found {len(subdirs)} subdirectories")
            
            for subdir in subdirs:
                pattern = os.path.join(subdir, f'*.{self.config.image_type}')
                image_count = len(glob.glob(pattern))
                
                if image_count >= self.config.batch_size:
                    logger.info(f"Processing subdirectory: {subdir} ({image_count} images)")
                    
                    subdir_name = os.path.basename(subdir)
                    output_subdir = os.path.join(self.config.output_dir, subdir_name)
                    
                    # Temporarily update output directory
                    original_output = self.config.output_dir
                    self.config.output_dir = output_subdir
                    
                    yield from self.batch_generator.from_directory(
                        subdir,
                        dead_dir=None,
                        live_dir=None
                    )
                    
                    # Restore original output directory
                    self.config.output_dir = original_output
                else:
                    logger.warning(f"Skipping {subdir}: only {image_count} images (minimum {self.config.batch_size})")
        else:
            # Process single directory
            logger.info(f"Processing images from {self.config.image_dir}")
            yield from self.batch_generator.from_directory(
                self.config.image_dir,
                dead_dir=None,
                live_dir=None
            )
    
    def _process_from_tfrecord(self) -> Generator:
        """Process images from TFRecord."""
        param_obj = param.Param()
        # Use center cropping instead of random cropping for consistent inference
        param_obj.randomcrop = False
        yield from self.batch_generator.from_tfrecord(param_obj, self.config.tfrecord_path)
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save results DataFrame to CSV."""
        from datetime import datetime
        output_name = Path(self.config.output_dir).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(self.config.output_dir) / f"{output_name}_results_{timestamp}.csv"
        
        try:
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
            
            # Log summary statistics
            if self.config.has_labels and 'saved' in results_df.columns:
                total = len(results_df)
                saved = results_df['saved'].sum()
                accuracy = (results_df['label'] == results_df['prediction']).sum() / total
                
                logger.info(f"Summary: {saved}/{total} visualizations saved")
                logger.info(f"Model accuracy: {accuracy:.2%}")
        except Exception as e:
            logger.error(f"Failed to save results CSV: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Grad-CAM visualizations for CNN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process images from directory
  python gradcam_refactored.py --im_dir /path/to/images --model_path model.keras --resdir /output
  
  # Process TFRecord with custom layer
  python gradcam_refactored.py --deploy_tfrec test.tfrecord --model_path model.keras \\
      --resdir /output --layer_name block5_conv3
  
  # Use layer preset for VGG16
  python gradcam_refactored.py --im_dir /path/to/images --model_path vgg16_model.keras \\
      --resdir /output --layer_preset vgg16
        """
    )
    
    # Input source (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--im_dir',
        help='Directory containing images to process'
    )
    input_group.add_argument(
        '--deploy_tfrec',
        help='Path to TFRecord file'
    )
    
    # Model configuration
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to trained Keras model (.keras or .h5)'
    )
    parser.add_argument(
        '--layer_name',
        default='conv5_block3_out',
        help='Target layer for Grad-CAM visualization (default: conv5_block3_out)'
    )
    parser.add_argument(
        '--layer_preset',
        choices=list(LAYER_PRESETS.keys()),
        help=f'Use preset layer name for architecture: {", ".join(LAYER_PRESETS.keys())}'
    )
    
    # Output configuration
    parser.add_argument(
        '--resdir',
        required=True,
        help='Output directory for visualizations'
    )
    
    # Processing options
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    parser.add_argument(
        '--imtype',
        default='tif',
        choices=['tif', 'tiff', 'jpg', 'jpeg', 'png'],
        help='Image file extension (default: tif)'
    )
    parser.add_argument(
        '--no_guided',
        action='store_true',
        help='Disable Guided Grad-CAM (use standard Grad-CAM)'
    )
    parser.add_argument(
        '--no_vgg_norm',
        action='store_true',
        help='Disable VGG normalization'
    )
    parser.add_argument(
        '--no_labels',
        action='store_true',
        help='Images do not have associated labels'
    )
    parser.add_argument(
        '--expected_label',
        type=int,
        choices=[0, 1],
        default=1,
        help='Expected label for all images in directory (0 or 1, default: 1)'
    )
    parser.add_argument(
        '--heatmap_only',
        action='store_true',
        help='Output only the Grad-CAM heatmap without overlaying on original image'
    )
    parser.add_argument(
        '--no_cropped',
        action='store_true',
        help='Disable saving cropped versions of original images'
    )
    parser.add_argument(
        '--target_size',
        nargs=2,
        type=int,
        default=[224, 224],
        metavar=('HEIGHT', 'WIDTH'),
        help='Target size for cropped images (height width, default: 224 224)'
    )
    parser.add_argument(
        '--orig_size',
        nargs=2,
        type=int,
        default=[512, 512],
        metavar=('HEIGHT', 'WIDTH'),
        help='Original size of input images (height width, default: 512 512)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Print banner
    banner = pyfiglet.figlet_format("GRADCAM", font="slant")
    print(banner)
    print(f"Version {__version__}\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine layer name
    layer_name = args.layer_name
    if args.layer_preset:
        layer_name = LAYER_PRESETS[args.layer_preset]
        logger.info(f"Using layer preset '{args.layer_preset}': {layer_name}")
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Layer: {layer_name}")
    logger.info(f"  Output: {args.resdir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Has labels: {not args.no_labels}")
    logger.info(f"  Save cropped: {not args.no_cropped}")
    logger.info(f"  Heatmap only: {args.heatmap_only}")
    logger.info(f"  Expected label: {args.expected_label}")
    
    try:
        # Create configuration
        config = GradCAMConfig(
            model_path=args.model_path,
            output_dir=args.resdir,
            image_dir=args.im_dir,
            tfrecord_path=args.deploy_tfrec,
            layer_name=layer_name,
            batch_size=args.batch_size,
            image_type=args.imtype,
            guided=not args.no_guided,
            vgg_normalize=not args.no_vgg_norm,
            has_labels=not args.no_labels,
            expected_label=args.expected_label,
            heatmap_only=args.heatmap_only,
            save_cropped=not args.no_cropped,
            target_size=tuple(args.target_size),
            orig_size=tuple(args.orig_size)
        )
        
        # Create processor and run
        processor = GradCAMProcessor(config)
        results = processor.run()
        
        logger.info(f"Processing complete! Results saved to {args.resdir}")
        logger.info(f"Processed {len(results)} images")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()