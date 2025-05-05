"""Utility to convert SVG to PNG

This module provides functionality to convert SVG files to PNG format
for use in PDF reports and other documents.
"""

import os
import io
import base64
from PIL import Image
import cairosvg

def svg_to_png(svg_path, output_path, width=300, height=100):
    """Convert SVG file to PNG
    
    Args:
        svg_path: Path to SVG file
        output_path: Path to save PNG file
        width: Output PNG width
        height: Output PNG height
        
    Returns:
        bool: True if conversion was successful
    """
    try:
        cairosvg.svg2png(url=svg_path, write_to=output_path, 
                        output_width=width, output_height=height)
        return True
    except Exception as e:
        print(f"Error converting SVG to PNG: {e}")
        return False

def convert_project_assets():
    """Convert project SVG assets to PNG format"""
    # Create assets directory if it doesn't exist
    os.makedirs('./assets', exist_ok=True)
    
    # Convert Arcadis logo
    if os.path.exists('./assets/arcadis_logo.svg'):
        svg_to_png('./assets/arcadis_logo.svg', './assets/arcadis_logo.png')
    
    # Convert business risk icon
    if os.path.exists('./assets/business_risk_icon.svg'):
        svg_to_png('./assets/business_risk_icon.svg', './assets/business_risk_icon.png', 
                  width=240, height=240)
        
    print("Asset conversion complete")

if __name__ == "__main__":
    convert_project_assets()
