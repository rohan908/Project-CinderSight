import argparse
import re
from typing import Dict, List, Optional, Text, Tuple
import pickle
import numpy as np
import torch
from pathlib import Path
import tfrecord

# Import shared configuration
from config import (
    ENHANCED_INPUT_FEATURES,
    NUM_ENHANCED_INPUT_FEATURES,
    OUTPUT_FEATURES,
    ENHANCED_DATA_STATS,
    DEFAULT_DATA_SIZE
)

"""Enhanced NDWS Dataset Cleaner and Preprocessor

This script processes the Enhanced Next Day Wildfire Spread (NDWS) dataset
with 19 input features as described in the paper by Rufai Yusuf Zakari et al.
Uses PyTorch for tensor operations.

Make sure to download the enhanced dataset TFRecord files and place them in the 
data directory before running this script.
"""



"""Library of common functions used in deep learning neural networks for Enhanced NDWS processing."""

def _get_base_key(key: Text) -> Text:
    """Extracts the base key from the provided key for normalization.

    Earth Engine exports TFRecords containing each data variable with its
    corresponding variable name. In the case of time sequences, the name of the
    data variable is of the form 'variable_1', 'variable_2', ..., 'variable_n',
    where 'variable' is the name of the variable, and n the number of elements
    in the time sequence. Extracting the base key ensures that each step of the
    time sequence goes through the same normalization steps.
    The base key obeys the following naming pattern: '([a-zA-Z]+)'
    For instance, for an input key 'variable_1', this function returns 'variable'.
    For an input key 'variable', this function simply returns 'variable'.

    Args:
        key: Input key.

    Returns:
        The corresponding base key.

    Raises:
        ValueError when `key` does not match the expected pattern.
    """
    match = re.match(r'([a-zA-Z]+)', key)
    if match:
        return match.group(1)
    raise ValueError(f'The provided key does not match the expected pattern: {key}')

def _clip_and_normalize(inputs: torch.Tensor, key: Text) -> torch.Tensor:
    """Clips and normalizes inputs with the stats corresponding to `key`.

    Args:
        inputs: Inputs to clip and normalize.
        key: Key describing the inputs.

    Returns:
        Clipped and normalized input.

    Raises:
        ValueError if there are no data statistics available for `key`.
    """
    base_key = _get_base_key(key)
    if base_key not in ENHANCED_DATA_STATS:
        raise ValueError(f'No data statistics available for the requested key: {key}.')
    
    min_val, max_val, mean, std = ENHANCED_DATA_STATS[base_key]
    inputs = torch.clamp(inputs, min_val, max_val)
    inputs = inputs - mean
    return inputs / std if std != 0 else inputs

def _parse_enhanced_example(
    record: Dict, 
    data_size: int, 
    sample_size: int,
    num_in_channels: int, 
    clip_and_normalize: bool,
    random_crop: bool, 
    center_crop: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reads a serialized example from Enhanced NDWS dataset.

    Args:
        record: A parsed TFRecord example dictionary.
        data_size: Size of tiles (square) as read from input files.
        sample_size: Size the tiles (square) when input into the model.
        num_in_channels: Number of input channels.
        clip_and_normalize: True if the data should be clipped and normalized.
        random_crop: True if the data should be randomly cropped.
        center_crop: True if the data should be cropped in the center.

    Returns:
        (input_img, output_img) tuple of inputs and outputs to the ML model.
    """
    
    if random_crop and center_crop:
        raise ValueError('Cannot have both random_crop and center_crop be True')
    
    input_features = ENHANCED_INPUT_FEATURES
    output_features = OUTPUT_FEATURES
    
    # Parse features from record
    parsed_features = {}
    for feature_name in input_features + output_features:
        if feature_name in record:
            # Convert from bytes/list to numpy array and reshape to data_size x data_size
            feature_data = np.array(record[feature_name]).reshape(data_size, data_size)
            parsed_features[feature_name] = torch.from_numpy(feature_data).float()
        else:
            # If feature not found, create zeros
            parsed_features[feature_name] = torch.zeros(data_size, data_size, dtype=torch.float32)

    # Process input features
    if clip_and_normalize:
        inputs_list = []
        for key in input_features:
            if key in ['PrevFireMask']:  # Don't normalize fire masks
                inputs_list.append(parsed_features[key])
            else:
                inputs_list.append(_clip_and_normalize(parsed_features[key], key))
    else:
        inputs_list = [parsed_features[key] for key in input_features]
    
    input_img = torch.stack(inputs_list, dim=0)  # Shape: (num_channels, H, W)
    input_img = input_img.permute(1, 2, 0)  # Shape: (H, W, num_channels)

    # Process output features
    outputs_list = [parsed_features[key] for key in output_features]
    output_img = torch.stack(outputs_list, dim=0)  # Shape: (1, H, W)
    output_img = output_img.permute(1, 2, 0)  # Shape: (H, W, 1)

    # Apply cropping if requested
    if random_crop:
        input_img, output_img = random_crop_input_and_output_images(
            input_img, output_img, sample_size, num_in_channels, 1)
    elif center_crop:
        input_img, output_img = center_crop_input_and_output_images(
            input_img, output_img, sample_size)
    
    return input_img, output_img

def random_crop_input_and_output_images(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    sample_size: int,
    num_in_channels: int,
    num_out_channels: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly axis-align crop input and output image tensors.

    Args:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        sample_size: side length (square) to crop to.
        num_in_channels: number of channels in input_img.
        num_out_channels: number of channels in output_img.

    Returns:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
    """
    combined = torch.cat([input_img, output_img], dim=2)
    
    # Random crop
    h, w, c = combined.shape
    start_h = torch.randint(0, h - sample_size + 1, (1,)).item()
    start_w = torch.randint(0, w - sample_size + 1, (1,)).item()
    
    combined = combined[start_h:start_h + sample_size, start_w:start_w + sample_size, :]
    
    input_img = combined[:, :, :num_in_channels]
    output_img = combined[:, :, -num_out_channels:]
    return input_img, output_img

def center_crop_input_and_output_images(
    input_img: torch.Tensor,
    output_img: torch.Tensor,
    sample_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Center crops input and output image tensors.

    Args:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
        sample_size: side length (square) to crop to.

    Returns:
        input_img: tensor with dimensions HWC.
        output_img: tensor with dimensions HWC.
    """
    h, w = input_img.shape[:2]
    start_h = (h - sample_size) // 2
    start_w = (w - sample_size) // 2
    
    input_img = input_img[start_h:start_h + sample_size, start_w:start_w + sample_size]
    output_img = output_img[start_h:start_h + sample_size, start_w:start_w + sample_size]
    return input_img, output_img

def get_enhanced_dataset(
    file_pattern: Text, 
    data_size: int, 
    sample_size: int,
    num_in_channels: int, 
    clip_and_normalize: bool, 
    random_crop: bool, 
    center_crop: bool
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Gets the Enhanced NDWS dataset from the file pattern.

    Args:
        file_pattern: Input file pattern for TFRecord files.
        data_size: Size of tiles (square) as read from input files.
        sample_size: Size the tiles (square) when input into the model.
        num_in_channels: Number of input channels.
        clip_and_normalize: True if the data should be clipped and normalized, False otherwise.
        random_crop: True if the data should be randomly cropped.
        center_crop: True if the data should be cropped in the center.

    Returns:
        A list of (input_img, output_img) tuples loaded from the input file pattern,
        with features described in the constants, and with the shapes determined from 
        the input parameters to this function.
    """
    
    from glob import glob
    tfrecord_files = glob(file_pattern)
    
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found matching pattern: {file_pattern}")
    
    dataset = []
    
    for tfrecord_file in tfrecord_files:
        try:
            # Use tfrecord library to read the file
            tfrecord_dataset = tfrecord.torch.TFRecordDataset(tfrecord_file, index_path=None)
            
            for record in tfrecord_dataset:
                try:
                    input_img, output_img = _parse_enhanced_example(
                        record, data_size, sample_size, num_in_channels, 
                        clip_and_normalize, random_crop, center_crop
                    )
                    dataset.append((input_img, output_img))
                except Exception as e:
                    print(f"Error parsing record in {tfrecord_file}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error reading TFRecord file {tfrecord_file}: {e}")
            continue
    
    return dataset

def remove_bad_samples(dataset: np.ndarray) -> np.ndarray:
    """Remove samples with missing data in target fire masks.
    
    Args:
        dataset: Dataset array with shape (N, C, H, W).
        
    Returns:
        Cleaned dataset with samples containing missing fire mask data removed.
    """
    fire_masks_array = np.array(dataset[:, -1, :, :])  # Last channel is FireMask
    good_indices = []

    print(f"Checking {len(fire_masks_array)} samples for missing data...")
    
    for img_num in range(len(fire_masks_array)):
        if np.all(fire_masks_array[img_num, :, :] != -1):
            good_indices.append(img_num)
        
        if (img_num + 1) % 1000 == 0:
            print(f"Processed {img_num + 1}/{len(fire_masks_array)} samples")

    print(f"Found {len(good_indices)} good samples out of {len(fire_masks_array)} total")
    print(f"Removed {len(fire_masks_array) - len(good_indices)} samples with missing data")
    
    return dataset[good_indices]

def analyze_dataset_statistics(dataset: np.ndarray):
    """Analyze and print enhanced dataset statistics.
    
    Args:
        dataset: Dataset array with shape (N, C, H, W).
    """
    print("\n" + "="*60)
    print("ENHANCED NDWS DATASET STATISTICS")
    print("="*60)
    
    n_samples, n_channels, height, width = dataset.shape
    print(f"Dataset shape: {dataset.shape}")
    print(f"Number of samples: {n_samples}")
    print(f"Spatial resolution: {height}x{width}")
    print(f"Number of channels: {n_channels}")
    
    # Separate features and target
    features = dataset[:, :-1, :, :]  # All except last channel
    fire_masks = dataset[:, -1, :, :]  # Last channel is FireMask
    
    # Fire distribution analysis
    fire_pixels = np.sum(fire_masks == 1)
    no_fire_pixels = np.sum(fire_masks == 0)
    unlabeled_pixels = np.sum(fire_masks == -1)
    total_pixels = n_samples * height * width
    
    print(f"\nFire Distribution:")
    print(f"Fire pixels: {fire_pixels:,} ({fire_pixels/total_pixels*100:.2f}%)")
    print(f"No fire pixels: {no_fire_pixels:,} ({no_fire_pixels/total_pixels*100:.2f}%)")
    print(f"Unlabeled pixels: {unlabeled_pixels:,} ({unlabeled_pixels/total_pixels*100:.2f}%)")
    
    # Feature statistics
    print(f"\nFeature Statistics:")
    for i, feature_name in enumerate(ENHANCED_INPUT_FEATURES):
        feature_data = features[:, i, :, :].flatten()
        valid_data = feature_data[feature_data != -1]
        
        if len(valid_data) > 0:
            print(f"{feature_name:12s}: "
                  f"mean={np.mean(valid_data):8.3f}, "
                  f"std={np.std(valid_data):8.3f}, "
                  f"min={np.min(valid_data):8.3f}, "
                  f"max={np.max(valid_data):8.3f}")
        else:
            print(f"{feature_name:12s}: No valid data found")
    
    print("="*60)

def validate_input_data_path(data_dir: str) -> Path:
    """Validate that the input data directory exists and contains required TFRecord files"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Input data directory does not exist: {data_path}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Input data path is not a directory: {data_path}")
    
    # Check for required TFRecord file patterns
    train_files = list(data_path.glob("*train*.tfrecord"))
    test_files = list(data_path.glob("*test*.tfrecord"))
    eval_files = list(data_path.glob("*eval*.tfrecord"))
    
    missing_files = []
    if not train_files:
        missing_files.append("*train*.tfrecord")
    if not test_files:
        missing_files.append("*test*.tfrecord")
    if not eval_files:
        missing_files.append("*eval*.tfrecord")
    
    if missing_files:
        raise FileNotFoundError(
            f"Missing required TFRecord files in {data_path}:\n"
            f"Missing patterns: {', '.join(missing_files)}\n"
            f"Please ensure you have training, test, and evaluation TFRecord files."
        )
    
    print(f"Validated input data directory: {data_path}")
    print(f"Found files:")
    print(f"  Training files: {len(train_files)}")
    print(f"  Test files: {len(test_files)}")
    print(f"  Evaluation files: {len(eval_files)}")
    
    return data_path

def create_output_data_path(output_dir: str) -> Path:
    """Create and validate output directory for processed data"""
    output_path = Path(output_dir)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions by creating a temporary file
        test_file = output_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()  # Remove test file
        except PermissionError:
            raise PermissionError(f"No write permission in output directory: {output_path}")
        
        print(f"Output directory ready: {output_path}")
        return output_path
        
    except PermissionError:
        raise PermissionError(f"Permission denied creating output directory: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_path}: {e}")

def main(data_dir: str = "data/raw", output_dir: str = "data/processed"):
    """Main processing function for Enhanced NDWS dataset with path parameters."""
    
    print("Enhanced NDWS Dataset Processor (PyTorch)")
    print("="*50)
    
    try:
        # Validate input data directory
        data_path = validate_input_data_path(data_dir)
        
        # Create and validate output directory
        output_path = create_output_data_path(output_dir)
        
        print(f"Input directory: {data_path}")
        print(f"Output directory: {output_path}")
        
        # Get file lists (we know they exist from validation)
        train_files = list(data_path.glob("*train*.tfrecord"))
        test_files = list(data_path.glob("*test*.tfrecord"))
        eval_files = list(data_path.glob("*eval*.tfrecord"))
        
        # Dataset parameters
        data_size = DEFAULT_DATA_SIZE  # Original data resolution
        sample_size = DEFAULT_DATA_SIZE  # Use full resolution
        num_in_channels = NUM_ENHANCED_INPUT_FEATURES  # 19 features
        
        print(f"\nProcessing parameters:")
        print(f"Data size: {data_size}x{data_size}")
        print(f"Sample size: {sample_size}x{sample_size}")
        print(f"Input channels: {num_in_channels}")
        
        # Process each dataset split
        datasets = {}
        
        for split_name, file_list in [("train", train_files), ("test", test_files), ("validation", eval_files)]:
            print(f"\n" + "="*30)
            print(f"Processing {split_name} dataset...")
            print("="*30)
            
            # Create file pattern
            file_pattern = str(data_path / f"*{split_name if split_name != 'validation' else 'eval'}*.tfrecord")
            
            try:
                # Create PyTorch dataset
                dataset = get_enhanced_dataset(
                    file_pattern=file_pattern,
                    data_size=data_size,
                    sample_size=sample_size,
                    num_in_channels=num_in_channels,
                    clip_and_normalize=True,
                    random_crop=False,  # Keep full size
                    center_crop=False
                )
                
                if not dataset:
                    print(f"Warning: No valid samples found for {split_name} dataset")
                    continue
                
                print(f"Successfully loaded {len(dataset)} samples!")
                
                # Convert to numpy arrays
                print("Converting to numpy arrays...")
                samples = []
                for i, (x, y) in enumerate(dataset):
                    if i % 1000 == 0:
                        print(f"Processing sample {i+1}/{len(dataset)}")
                    
                    # Convert PyTorch tensors to numpy and combine
                    x_np = x.numpy()  # Shape: (H, W, C)
                    y_np = y.numpy()  # Shape: (H, W, 1)
                    combined = np.concatenate([x_np, y_np], axis=2)
                    samples.append(combined)
                
                # Stack all samples and transpose to (N, C, H, W) format
                dataset_array = np.moveaxis(np.array(samples), 3, 1)
                print(f"Dataset shape: {dataset_array.shape}")
                
                # Remove bad samples
                dataset_clean = remove_bad_samples(dataset_array)
                
                # Store the processed dataset
                datasets[split_name] = dataset_clean
                
                # Analyze statistics
                analyze_dataset_statistics(dataset_clean)
                
            except Exception as e:
                print(f"Error processing {split_name} dataset: {e}")
                print("You may need to install: pip install tfrecord torch")
                continue
        
        if not datasets:
            raise RuntimeError("No datasets were successfully processed")
        
        # Save processed datasets
        print("\n" + "="*30)
        print("Saving processed datasets...")
        print("="*30)
        
        for split_name, dataset in datasets.items():
            # Separate features and labels
            features = dataset[:, :-1, :, :]  # All channels except last
            labels = dataset[:, -1, :, :]     # Last channel (FireMask)
            
            # Save features
            features_path = output_path / f"{split_name}.data"
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)
            print(f"Saved {split_name} features: {features.shape} -> {features_path}")
            
            # Save labels
            labels_path = output_path / f"{split_name}.labels"
            with open(labels_path, 'wb') as f:
                pickle.dump(labels, f)
            print(f"Saved {split_name} labels: {labels.shape} -> {labels_path}")
        
        # Save feature names and statistics
        metadata = {
            'input_features': ENHANCED_INPUT_FEATURES,
            'output_features': OUTPUT_FEATURES,
            'data_stats': ENHANCED_DATA_STATS,
            'data_size': data_size,
            'sample_size': sample_size,
            'num_input_channels': num_in_channels,
            'framework': 'pytorch'
        }
        
        metadata_path = output_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Saved metadata -> {metadata_path}")
        
        print(f"\nEnhanced NDWS dataset processing completed successfully!")
        print(f"Processed datasets saved in: {output_path}")
        print(f"Total samples processed:")
        for split_name, dataset in datasets.items():
            print(f"  {split_name}: {len(dataset)} samples")
        
        return True
        
    except FileNotFoundError as e:
        print(f"File/Directory Error: {e}")
        print("Please ensure:")
        print(f"  1. Input data directory exists: {data_dir}")
        print("  2. Required TFRecord files are present")
        return False
    except PermissionError as e:
        print(f"Permission Error: {e}")
        print("Please check directory permissions")
        return False
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
        print("You may need to install: pip install tfrecord torch")
        print(f"Input directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced NDWS Dataset Processor (PyTorch)')
    parser.add_argument('--data_dir', '-d', default='data/raw',
                        help='Directory containing TFRecord files')
    parser.add_argument('--output_dir', '-o', default='data/processed',
                        help='Directory to save processed datasets')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir) 