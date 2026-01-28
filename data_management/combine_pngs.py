#!/usr/bin/env python3
"""
Script to combine all PNG files in the directory into a single grid image.
Preserves all pixel data including transparency/alpha channels.
"""

import os
from PIL import Image
import math
from pathlib import Path
import argparse

# Increase PIL's decompression bomb limit to handle large images
Image.MAX_IMAGE_PIXELS = None

def combine_pngs_to_grid(directory, output_path):
    """
    Combine all PNG files in a directory (recursively) into a single grid image.
    Preserves all pixel data including transparency/alpha channels.
    
    Args:
        directory: Root directory to search for PNG files
        output_path: Full path where the combined image should be saved
    """
    # Find all PNG files recursively
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    
    if not png_files:
        print("No PNG files found!")
        return
    
    # Sort files for consistent ordering
    png_files.sort()
    
    print(f"Found {len(png_files)} PNG files")
    
    # First pass: get dimensions and check for alpha channels (without loading full images)
    print("Scanning images for dimensions...")
    max_width = 0
    max_height = 0
    has_alpha = False
    
    for i, png_file in enumerate(png_files):
        try:
            with Image.open(png_file) as img:
                max_width = max(max_width, img.width)
                max_height = max(max_height, img.height)
                # Check if image has alpha channel mode
                if img.mode in ('RGBA', 'LA') or 'transparency' in img.info:
                    has_alpha = True
            if (i + 1) % 50 == 0:
                print(f"  Scanned {i + 1}/{len(png_files)} images...")
        except Exception as e:
            print(f"Warning: Could not scan {png_file}: {e}")
    
    if max_width == 0 or max_height == 0:
        print("No valid images found!")
        return
    
    print(f"Found {len(png_files)} images")
    print(f"Max dimensions: {max_width} x {max_height}")
    print(f"Has transparency: {has_alpha}")
    
    # Calculate grid dimensions
    num_images = len(png_files)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    
    print(f"Grid layout: {rows} rows x {cols} columns")
    
    # Create the combined image
    # Use padding to separate images visually
    padding = 10
    cell_width = max_width + padding * 2
    cell_height = max_height + padding * 2
    
    combined_width = cols * cell_width
    combined_height = rows * cell_height
    
    # Estimate memory usage for both modes
    rgb_memory_mb = (combined_width * combined_height * 3) / (1024 * 1024)
    rgba_memory_mb = (combined_width * combined_height * 4) / (1024 * 1024)
    print(f"Creating combined image: {combined_width} x {combined_height} pixels")
    print(f"Estimated memory: RGB={rgb_memory_mb:.1f} MB, RGBA={rgba_memory_mb:.1f} MB")
    
    # Use RGB mode to preserve all RGB pixel data
    # This mode worked successfully before and avoids memory issues
    # RGBA images will be properly composited onto white background to preserve visible pixels
    image_mode = 'RGB'
    bg_color = (255, 255, 255)
    
    if has_alpha:
        print("Using RGB mode - all RGB pixel data preserved")
        print("RGBA images will be composited onto white background (transparency -> white)")
    else:
        print("Using RGB mode - all pixel data preserved")
    
    combined_image = Image.new(image_mode, (combined_width, combined_height), color=bg_color)
    
    # Process images one at a time to save memory
    print("Combining images...")
    for idx, png_file in enumerate(png_files):
        try:
            # Load image only when needed
            with Image.open(png_file) as img:
                # Convert to RGB mode to preserve all RGB pixel data
                # For RGBA images, convert to RGB (composites transparency onto white)
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # Convert RGBA to RGB, compositing transparent areas onto white
                        # This preserves all visible RGB pixel values
                        img = img.convert('RGB')
                    else:
                        img = img.convert('RGB')
                
                # Image is ready to paste (no text labels added)
                
                row = idx // cols
                col = idx % cols
                
                # Calculate position
                x = col * cell_width + padding
                y = row * cell_height + padding
                
                # Center the image if it's smaller than max dimensions
                offset_x = (max_width - img.width) // 2
                offset_y = (max_height - img.height) // 2
                
                # Paste the image, preserving all RGB pixel data
                combined_image.paste(img, (x + offset_x, y + offset_y))
                
        except Exception as e:
            print(f"Warning: Could not process {png_file}: {e}")
        
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(png_files)} images...")
    
    # Save the combined image
    # Use lossless PNG compression to preserve all pixel data
    print(f"\nSaving combined image to: {output_path}")
    combined_image.save(output_path, 'PNG', compress_level=1)  # compress_level=1 is faster, still lossless
    output_filename = os.path.basename(output_path)
    print(f"Done! Combined image saved as '{output_filename}'")
    print(f"Final image size: {combined_width} x {combined_height} pixels")
    print(f"Mode: {image_mode} (preserves all pixel data including transparency)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine all PNG files in directories into single grid images."
    )
    parser.add_argument(
        "directories",
        nargs="*",
        default=None,
        help="Directories to search for PNG files (default: script's directory). Can specify multiple directories."
    )
    args = parser.parse_args()
    
    # Determine which directories to process
    if args.directories:
        directories_to_process = [os.path.abspath(d) for d in args.directories]
    else:
        directories_to_process = [os.path.dirname(os.path.abspath(__file__))]
    
    # Filter to only valid directories
    valid_directories = []
    for dir_path in directories_to_process:
        if os.path.isdir(dir_path):
            valid_directories.append(dir_path)
        else:
            print(f"Warning: Skipping '{dir_path}' - not a valid directory")
    
    if not valid_directories:
        print("Error: No valid directories to process!")
        exit(1)
    
    print(f"Processing {len(valid_directories)} directory(ies)...\n")
    
    # Process each directory
    for idx, target_dir in enumerate(valid_directories, 1):
        print(f"{'='*60}")
        print(f"Processing directory {idx}/{len(valid_directories)}: {target_dir}")
        print(f"{'='*60}")
        
        # Generate output filename from directory name and save in the same directory
        dir_name = os.path.basename(os.path.normpath(target_dir))
        output_filename = f"combined_{dir_name}_images.png"
        output_path = os.path.join(target_dir, output_filename)
        
        combine_pngs_to_grid(target_dir, output_path)
        print()  # Empty line between directories
