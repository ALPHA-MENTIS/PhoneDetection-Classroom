import cv2
import numpy as np

def calculate_entropy(image_roi):
    """
    Calculates the Shannon Entropy of an image region.
    
    Concept:
    Entropy measures the amount of "randomness" or "information" in an image.
    - Scientific Calculators: High Entropy. They have many buttons, text, grids, and complex textures.
    - Phones: Low Entropy. Screens (even when displaying content) usually have large smooth areas.
    
    Match:
    Entropy = -sum(p * log2(p)) where p is the probability of a pixel intensity.
    """
    # Convert to grayscale if not already
    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi
        
    # Calculate histogram of pixel intensities (0-255)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Normalize histogram to get probabilities
    # total_pixels = gray.size
    hist_norm = hist.ravel() / hist.sum()
    
    # Calculate entropy
    # Filter out 0 values to avoid log(0) error
    hist_norm = hist_norm[hist_norm > 0]
    entropy = - (hist_norm * np.log2(hist_norm)).sum()
    
    return entropy

def detect_specular_highlight(image_roi, threshold=200):
    """
    Detects specular highlights (glare) which are characteristic of glass screens.
    
    Concept:
    - Phones: Glass screens reflect light sources sharply (specular reflection).
    - Calculators: Plastic buttons/bodies diffuse light (diffuse reflection).
    
    Returns:
    - score (float): A score indicating likelihood of glass reflection.
    - has_glare (bool): True if significant glare is found.
    """
    # Convert to grayscale
    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi
        
    # Apply a binary threshold to find very bright spots
    # Pixels > threshold (e.g., 200) are considered potential highlights
    _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Count the bright pixels
    num_bright_pixels = cv2.countNonZero(bright_mask)
    total_pixels = gray.size
    
    # Calculate ratio of bright pixels
    bright_ratio = num_bright_pixels / total_pixels
    
    # Heuristic:
    # If there are small, concentrated bright spots, it's likely glass.
    # If the whole image is bright (e.g., a white piece of paper), the ratio might be high but it's not a "highlight".
    # (For now, we'll stick to a simple density check, but we could add contour analysis later)
    
    return bright_ratio, bright_ratio > 0.01  # e.g., if >1% of pixels are "glare"
