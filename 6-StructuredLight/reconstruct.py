# ================================================
#    Structured Light 
# ================================================

import cv2
import sys
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region

    h,w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))
    #print proj_mask
    scan_bits = np.zeros((h,w), dtype=np.uint16)
    '''
    print scan_bits
    output_name = "./" + "test.png"
    cv2.imwrite(output_name, scan_bits);
    '''

    # analyze the binary patterns from the camera
    for i in range(0,15):
        # read the file
        patt = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2)), (0,0), fx=scale_factor,fy=scale_factor)
        #patt_gray = cv2.cvtColor(patt, cv2.COLOR_BGR2GRAY)/255.0
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2), cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask
        #print on_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        print bit_code
        
        # TODO: populate scan_bits by putting the bit_code according to on_mask
        for i in range(h):
            for j in range(w):
                if on_mask[i][j] == True :                    
                #if [j for i in range(w) for j in range(h)] == True:
                    scan_bits[i][j] = scan_bits[i][j] | bit_code
                else:
                    continue
    #scan_bits = cv2.bitwise_and(scan_bits, on_mask)
    '''
    print scan_bits
    #print scan_bits`
    output_name = "./" + "test.png"
    cv2.imwrite(output_name, scan_bits);
    '''

    print("load codebook")
    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    data = np.zeros((h,w,3), dtype=np.float)

    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
                #projector_points[y/2][x/2] = proj_mask
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            p_x, p_y = binary_codes_ids_codebook[scan_bits[y][x]]
            if p_x >= 1279 or p_y >= 799: # filter
                continue

            data[y][x] = [0,p_y*255/799.0,p_x*255/1279.0]
            # due to differences in calibration and acquisition - divide the camera points by 2
    
            #Store the points in camera_points and projector_points
            #projector_points.append([[p_x][p_y]])
            #camera_points.append([[x/2][y/2]])

    cv2.imwrite("correspondance.jpg", data)
    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
	# TODO: name the resulted 3D points as "points_3d"
	
    return points_3d
	
def write_3d_points(points_3d):
	
	# ===== DO NOT CHANGE THIS FUNCTION =====
    print ("write output point cloud")
    print (points_3d.shape) 
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0,0],p[0,1],p[0,2]))

    return points_3d, camera_points, projector_points
    
if __name__ == '__main__':

	# ===== DO NOT CHANGE THIS FUNCTION =====
	
	# validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
	
