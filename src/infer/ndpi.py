"""
NDPI Processor Module

This module provides functionality to extract and process NDPI (NanoZoomer Digital Pathology Image) files.
It can detect individual samples within an NDPI file and extract them as separate TIF files.
"""

import os
import numpy as np
import cv2
from PIL import Image
from typing import List, Tuple, Optional

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    print("Warning: openslide-python not available. NDPI support will be limited.")


class NDPIExtractor:
    """Extract and convert NDPI files to TIF format, splitting into individual samples using OpenSlide."""
    
    def __init__(self, input_file: str, output_dir: Optional[str] = None):
        """
        Initialize the NDPI extractor.
        
        Args:
            input_file: Path to the input NDPI file
            output_dir: Directory to save extracted TIF files (defaults to same directory as input)
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("openslide-python is required for NDPI support. Install with: pip install openslide-python")
        self.input_file = input_file
        self.output_dir = output_dir if output_dir else os.path.dirname(input_file)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_sample_regions(self, image_array: np.ndarray, debug: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        Detect sample regions in the NDPI image by analyzing white column separators between samples.
        Samples are typically arranged horizontally with vertical white space between them.
        
        Args:
            image_array: Numpy array of the NDPI image (RGB format)
            debug: If True, print debug information
            
        Returns:
            List of (x1, y1, x2, y2) tuples representing sample regions
        """
        try:
            h, w = image_array.shape[:2]
            
            if debug:
                print(f"Image dimensions: {w}x{h}")
            
            # Calculate column averages across all channels
            # Shape: (width, channels)
            column_averages = np.mean(image_array, axis=0)
            
            # Calculate the overall brightness of each column (average of RGB)
            column_brightness = np.mean(column_averages, axis=1)
            
            if debug:
                print(f"Column brightness - min: {column_brightness.min():.2f}, max: {column_brightness.max():.2f}, mean: {column_brightness.mean():.2f}")
            
            # Use adaptive threshold based on the image statistics
            # Find columns that are significantly brighter than the median
            brightness_threshold = np.percentile(column_brightness, 85)  # Top 15% brightest
            
            # A column is "white" if it's in the top 15% brightest
            white_columns = column_brightness >= brightness_threshold
            
            # Find non-white (significant) columns
            significant_columns = np.where(~white_columns)[0]
            
            if debug:
                print(f"Brightness threshold (85th percentile): {brightness_threshold:.2f}")
                print(f"Total columns: {w}")
                print(f"White columns: {np.sum(white_columns)} ({100*np.sum(white_columns)/w:.2f}%)")
                print(f"Significant columns: {len(significant_columns)}")
            
            if len(significant_columns) == 0:
                print("Warning: No significant content detected, using full image")
                return [(0, 0, w, h)]
            
            # Find continuous non-white regions (samples)
            # Use a smaller minimum gap size to detect narrower separators
            min_gap_size = max(5, w // 500)  # At least 5 pixels or 0.2% of width
            
            non_white_regions = []
            region_start = significant_columns[0]
            gaps_found = []
            
            for i in range(1, len(significant_columns)):
                gap_size = significant_columns[i] - significant_columns[i - 1] - 1
                
                # Track all gaps for debugging
                if gap_size > 0 and debug:
                    gaps_found.append((significant_columns[i - 1], significant_columns[i], gap_size))
                
                # If there's a significant gap in columns, we found a separator
                if gap_size >= min_gap_size:
                    # End the current region
                    non_white_regions.append((region_start, significant_columns[i - 1]))
                    # Start a new region
                    region_start = significant_columns[i]
            
            # Add the last region
            non_white_regions.append((region_start, significant_columns[-1]))
            
            if debug:
                print(f"Minimum gap size for separation: {min_gap_size} pixels")
                
                # Show largest gaps found
                if gaps_found:
                    gaps_found.sort(key=lambda x: x[2], reverse=True)
                    print(f"Top 10 largest gaps found:")
                    for i, (start, end, size) in enumerate(gaps_found[:10], 1):
                        status = "✓ SEPARATOR" if size >= min_gap_size else "✗ too small"
                        print(f"  {i}. Gap at columns {start}-{end}: {size} pixels {status}")
                
                print(f"Detected {len(non_white_regions)} sample regions by column analysis")
                for i, (start, end) in enumerate(non_white_regions, 1):
                    print(f"  Region {i}: columns {start} to {end} (width: {end - start + 1})")
            
            # Filter out very small regions (likely noise)
            min_region_width = max(100, w // 50)  # At least 100 pixels or 2% of width
            filtered_regions = [(start, end) for start, end in non_white_regions 
                               if (end - start + 1) >= min_region_width]
            
            if debug and len(filtered_regions) < len(non_white_regions):
                print(f"Filtered out {len(non_white_regions) - len(filtered_regions)} small regions (< {min_region_width} pixels)")
                print(f"Remaining regions: {len(filtered_regions)}")
            
            # If no regions after filtering, use the original regions
            if not filtered_regions:
                filtered_regions = non_white_regions
            
            # Convert column ranges to bounding boxes
            # For each region, find the vertical extent of non-white content and remove horizontal white areas
            regions = []
            for region_idx, (x_start, x_end) in enumerate(filtered_regions, 1):
                # Extract the region
                region = image_array[:, x_start:x_end+1, :]
                
                # Calculate row brightness (average across width and channels)
                row_brightness = np.mean(region, axis=(1, 2))
                
                # Use a more aggressive threshold for horizontal trimming
                # We want to remove clearly white/bright rows at top and bottom
                row_brightness_threshold = np.percentile(row_brightness, 75)  # Top 25% brightest
                
                # Find non-white rows (rows below the brightness threshold)
                non_white_rows = np.where(row_brightness < row_brightness_threshold)[0]
                
                if debug:
                    print(f"  Region {region_idx}: Row brightness range: {row_brightness.min():.2f}-{row_brightness.max():.2f}, threshold: {row_brightness_threshold:.2f}")
                
                if len(non_white_rows) > 0:
                    # Find continuous non-white regions in vertical direction
                    # This handles cases where there might be horizontal white bands within the sample
                    min_vertical_gap = max(5, h // 200)  # Minimum gap to split vertically
                    
                    vertical_regions = []
                    v_region_start = non_white_rows[0]
                    
                    for i in range(1, len(non_white_rows)):
                        v_gap_size = non_white_rows[i] - non_white_rows[i - 1] - 1
                        
                        # If there's a significant vertical gap, split here
                        if v_gap_size >= min_vertical_gap:
                            vertical_regions.append((v_region_start, non_white_rows[i - 1]))
                            v_region_start = non_white_rows[i]
                    
                    # Add the last vertical region
                    vertical_regions.append((v_region_start, non_white_rows[-1]))
                    
                    # Use the largest vertical region (main content)
                    if vertical_regions:
                        # Sort by size and take the largest
                        vertical_regions.sort(key=lambda r: r[1] - r[0], reverse=True)
                        y_start, y_end = vertical_regions[0]
                        
                        if debug and len(vertical_regions) > 1:
                            print(f"  Region {region_idx}: Found {len(vertical_regions)} vertical segments, using largest")
                    else:
                        y_start = non_white_rows[0]
                        y_end = non_white_rows[-1]
                else:
                    # If no non-white rows found, use full height
                    y_start = 0
                    y_end = h - 1
                
                # Add padding (more horizontal padding to include slightly more area vertically)
                pad_x = 60  # Increased horizontal padding
                pad_y = 5  # Keep vertical padding small
                x1 = max(0, x_start - pad_x)
                y1 = max(0, y_start - pad_y)
                x2 = min(w, x_end + pad_x)
                y2 = min(h, y_end + pad_y)
                
                regions.append((x1, y1, x2, y2))
            
            if debug:
                print(f"Final bounding boxes (after removing horizontal white areas):")
                for i, (x1, y1, x2, y2) in enumerate(regions, 1):
                    print(f"  Sample {i}: ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
            
            return regions
            
        except Exception as e:
            print(f"Warning: Error detecting sample regions: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fallback to whole image if detection fails
            return [(0, 0, image_array.shape[1], image_array.shape[0])]
    
    def extract_to_tif(self, compression: str = "lzw", level: int = 2, debug: bool = False) -> List[str]:
        """
        Extract NDPI file to multiple TIF files, one per sample.
        
        Args:
            compression: Compression method for TIF output (default: "lzw")
            level: Pyramid level to extract (default: 2, same as training)
            debug: If True, print debug information
            
        Returns:
            List of paths to the extracted TIF files
        """
        try:
            # Open NDPI file with OpenSlide
            slide = openslide.OpenSlide(self.input_file)
            
            # Get the dimensions at the specified level
            if level >= len(slide.level_dimensions):
                print(f"Warning: Level {level} not available, using level 0")
                level = 0
            
            width, height = slide.level_dimensions[level]
            print(f"Extracting NDPI at level {level}: {width}x{height}")
            
            # Read the region at the specified level
            image = slide.read_region((0, 0), level, (width, height))
            
            # Convert RGBA to RGB and to numpy array
            image_array = np.array(image.convert('RGB'))
            
            # Detect sample regions with debug output
            print("Analyzing NDPI image to detect individual samples...")
            sample_regions = self.detect_sample_regions(image_array, debug=debug)
            print(f"Detected {len(sample_regions)} sample regions")
            
            # Print region details
            for i, (x1, y1, x2, y2) in enumerate(sample_regions, 1):
                print(f"  Sample {i}: region ({x1}, {y1}) to ({x2}, {y2}), size: {x2-x1}x{y2-y1}")
            
            output_paths = []
            base_name = os.path.splitext(os.path.basename(self.input_file))[0]
            
            for i, (x1, y1, x2, y2) in enumerate(sample_regions, 1):
                try:
                    # Extract the sample region
                    sample = image.crop((x1, y1, x2, y2))
                    sample_array = np.array(sample)
                    
                    # Trim white borders
                    gray = (0.299 * sample_array[:, :, 0] + 
                           0.587 * sample_array[:, :, 1] + 
                           0.114 * sample_array[:, :, 2]).astype(np.uint8)
                    
                    non_white_mask = gray < 250
                    if np.any(non_white_mask):
                        rows = np.any(non_white_mask, axis=1)
                        cols = np.any(non_white_mask, axis=0)
                        y_min, y_max = np.where(rows)[0][[0, -1]]
                        x_min, x_max = np.where(cols)[0][[0, -1]]
                        sample_array = sample_array[y_min:y_max+1, x_min:x_max+1]
                    
                    # Create output filename
                    output_name = f"{base_name}_sample_{i:02d}.tif"
                    output_path = os.path.join(self.output_dir, output_name)
                    
                    # Save as TIF using PIL with LZW compression
                    Image.fromarray(sample_array).save(output_path, compression='tiff_lzw')
                    print(f"Saved sample {i} to: {output_path}")
                    output_paths.append(output_path)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {str(e)}")
                    continue
            
            slide.close()
            
            if not output_paths:
                raise ValueError("No valid samples were extracted from the NDPI file")
                
            return output_paths
            
        except Exception as e:
            print(f"Error extracting NDPI file: {str(e)}")
            if 'slide' in locals():
                slide.close()
            raise


def main():
    """Test the NDPI extractor with a sample file."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract samples from NDPI files')
    parser.add_argument('input_file', help='Path to input NDPI file')
    parser.add_argument('--output-dir', '-o', help='Output directory (default: same as input)', default=None)
    parser.add_argument('--level', '-l', type=int, default=2, help='Pyramid level to extract (default: 2)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--visualize', '-v', action='store_true', help='Save visualization of detected regions')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)
    
    print(f"Processing NDPI file: {args.input_file}")
    print(f"Level: {args.level}")
    print(f"Debug: {args.debug}")
    print("-" * 80)
    
    try:
        # Create extractor
        extractor = NDPIExtractor(args.input_file, args.output_dir)
        
        # If visualization requested, load the image and show detection
        if args.visualize:
            print("\nLoading image for visualization...")
            slide = openslide.OpenSlide(args.input_file)
            
            # Get the dimensions at the specified level
            level = args.level
            if level >= len(slide.level_dimensions):
                print(f"Warning: Level {level} not available, using level 0")
                level = 0
            
            width, height = slide.level_dimensions[level]
            print(f"Image dimensions at level {level}: {width}x{height}")
            
            # Read the region
            image = slide.read_region((0, 0), level, (width, height))
            image_array = np.array(image.convert('RGB'))
            
            # Detect regions
            print("\nDetecting sample regions...")
            regions = extractor.detect_sample_regions(image_array, debug=True)
            
            # Create visualization
            print("\nCreating visualization...")
            vis_image = image_array.copy()
            
            # Draw rectangles on the visualization
            for i, (x1, y1, x2, y2) in enumerate(regions, 1):
                # Draw rectangle
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=max(2, width // 1000))
                
                # Add label
                label = f"Sample {i}"
                font_scale = max(1.0, width / 5000)
                thickness = max(2, width // 2000)
                cv2.putText(vis_image, label, (x1 + 20, y1 + 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
            
            # Save visualization
            output_dir = args.output_dir if args.output_dir else os.path.dirname(args.input_file)
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.input_file))[0]
            vis_path = os.path.join(output_dir, f"{base_name}_detection_visualization.png")
            
            # Resize if too large for viewing
            max_dim = 4000
            if max(vis_image.shape[:2]) > max_dim:
                scale = max_dim / max(vis_image.shape[:2])
                new_width = int(vis_image.shape[1] * scale)
                new_height = int(vis_image.shape[0] * scale)
                vis_image = cv2.resize(vis_image, (new_width, new_height))
            
            cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"Saved visualization to: {vis_path}")
            
            slide.close()
        
        # Extract samples
        print("\n" + "=" * 80)
        print("Extracting samples to TIF files...")
        print("=" * 80)
        output_files = extractor.extract_to_tif(level=args.level, debug=args.debug)
        
        print("\n" + "=" * 80)
        print(f"Successfully extracted {len(output_files)} samples:")
        for i, path in enumerate(output_files, 1):
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            print(f"  {i}. {os.path.basename(path)} ({file_size:.2f} MB)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
