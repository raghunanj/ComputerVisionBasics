 Goal is to perform semi-automatic binary segmentation based on SLIC superpixels and graph-cuts:
 
 Given an image and sparse markings for foreground and background.
 Approach followed with the steps:
 
 1. Calculate SLIC over image.
 2. Calculate color histograms for all superpixels.
 3. Calculate color histograms for Foreground and background.
 4. Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
 5. Run a graph-cut algorithm to get the final segmentation.
 
 The repo consists of 2 separate files, `main.py` and `main_bonus.py`.
 
 `main.py` has the above described functionality.
 
 `main_bonus.py` has the added functionality where the user describes the FG/BG mask by interacting with the Console using opencv function cv.setMouseCallback(). 
 `bonus.mov` has the recorded video of a sample working session.
 
 
