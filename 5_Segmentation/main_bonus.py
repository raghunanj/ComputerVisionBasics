import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

def help_message():
   print("Usage: [Input_Image] [Input_Marking] [Output_Directory]")
   print("[Input_Image]")
   print("Path to the input image")
   print("[Input_Marking]")
   print("Path to the input marking")
   print("[Output_Directory]")
   print("Output directory")
   print("Example usages:")
   print(sys.argv[0] + " astronaut.png " + "astronaut_marking.png " + "./")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=500, compactness=18.5)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

#Functionality for finding out the normalized graph cut segmented image.
def Segmentation(img_marking, superpixels, color_hists, neighbors):
    #centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

    fgbg_hists = [fg_cumulative_hist,bg_cumulative_hist]
    fgbg_superpixels = [fg_segments,bg_segments]
    
    norm_hists = normalize_histograms(color_hists)

    graph_cut = do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors)

    segment_mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))

    mask = np.uint8(segment_mask * 255)

    cv2.imshow('mask', mask)

    cv2.moveWindow('mask',300,25)


#Interactive session code for the BG/FG markings
def interactive_drawing(event, x, y, flags, param):
    global ix, iy, drawing, left_button, counterL, counterR

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        left_button = True
        counterL = 1
        if (counterL + counterR == 2):
            Segmentation(img_marking, superpixels, color_hists, neighbors)
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        left_button = False
        counterR = 1
        if (counterL + counterR == 2):
            Segmentation(img_marking, superpixels, color_hists, neighbors)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if left_button == True:
                cv2.line(img_marking, (ix, iy), (x, y), (0, 0, 255), 10)
                cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 10)
                ix, iy = x, y
            if left_button == False:
                cv2.line(img_marking, (ix, iy), (x, y), (255, 0, 0), 10)
                cv2.line(img, (ix, iy), (x, y), (255, 0, 0), 10)
                ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if left_button == True:
            cv2.line(img_marking, (ix, iy), (x, y), (0, 0, 255), 10)
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 10)
            ix, iy = x, y
        if left_button == False:
            cv2.line(img_marking, (ix, iy), (x, y), (255, 0, 0), 10)
            cv2.line(img, (ix, iy), (x, y), (255, 0, 0), 10)
            ix, iy = x, y
        counterL = 1
        if (counterL + counterR == 2):
            Segmentation(img_marking, superpixels, color_hists, neighbors)
    elif event == cv2.EVENT_RBUTTONUP:
        drawing = False
        if left_button == True:
            cv2.line(img_marking, (ix, iy), (x, y), (0, 0, 255), 10)
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 10)
            ix, iy = x, y
        if left_button == False:
            cv2.line(img_marking, (ix, iy), (x, y), (255, 0, 0), 10)
            cv2.line(img, (ix, iy), (x, y), (255, 0, 0), 10)
            ix, iy = x, y
        counterR = 1
        if (counterL + counterR == 2):
            Segmentation(img_marking, superpixels, color_hists, neighbors)
    return x, y

if __name__ == '__main__':
   
    # validate the input arguments
    if (len(sys.argv) >= 4):
        help_message()
        sys.exit()

    img = cv2.imread(sys.argv[1], 1)

    # ======================================== #
    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img)
    
    img_marking = np.zeros(img.shape, np.uint8)
    img_marking[:] = (255, 255, 255)
    #initial global declarations
    drawing = False
    left_button = True
    counterL = 0
    counterR = 0

    #Iterative function for the interactive mouse input session
    cv2.namedWindow('Markings')
    cv2.setMouseCallback('Markings', interactive_drawing)
    while(1):
        cv2.imshow('Markings', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC button
            break
    cv2.destroyAllWindows()

    # ======================================== #
