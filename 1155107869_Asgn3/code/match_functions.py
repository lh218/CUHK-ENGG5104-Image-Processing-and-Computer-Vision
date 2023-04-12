import numpy as np
import cv2

def get_interest_points(image, feature_width):
    """ Returns a set of interest points for the input image
    Args:
        image - can be grayscale or color, your choice.
        feature_width - in pixels, is the local feature width. It might be
            useful in this function in order to (a) suppress boundary interest
            points (where a feature wouldn't fit entirely in the image)
            or (b) scale the image filters being used. Or you can ignore it.
    Returns:
        x and y: nx1 vectors of x and y coordinates of interest points.
        confidence: an nx1 vector indicating the strength of the interest
            point. You might use this later or not.
        scale and orientation: are nx1 vectors indicating the scale and
            orientation of each interest point. These are OPTIONAL. By default you
            do not need to make scale and orientation invariant local features. 
    """
    
    # Use cv2.cornerHarris to find the corners
    output = cv2.cornerHarris(image, 2, 3, 0.05)
    
    # Process
    _, output = cv2.threshold(output, 0.01 * output.max(), 255, 0)
    output = np.uint8(output)
    _, _, _, centroids = cv2.connectedComponentsWithStats(output)
    
    # Parameters for cv2.cornerSubPix
    corner = np.float32(centroids)
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    
    corners = cv2.cornerSubPix(image, corner, winSize, zeroZone, criteria)
    x, y = corners[:, 0:1], corners[:, 1:2]
    
    return x, y

    # Personal Implementation of Harris Corner (with less efficiency)
    
    # corner_x = []
    # corner_y = []
    # k = 0.04
    # threshold = 1
    
    # h, w = image.shape[:2]
    # window_bound = int(feature_width/2)
    # y_bound = h - window_bound
    # x_bound = w - window_bound
    
    # dy, dx = np.gradient(image)
    # Ixx = dx**2
    # Ixy = dy*dx
    # Iyy = dy**2
    
    # for y in range(window_bound, y_bound):
    #     for x in range(window_bound, x_bound):
    #         y_1 = y - window_bound
    #         y_2 = y + window_bound + 1
    #         x_1 = x - window_bound
    #         x_2 = x + window_bound + 1
            
    #         Ixx_window = Ixx[y_1 : y_2, x_1 : x_2]
    #         Ixy_window = Ixy[y_1 : y_2, x_1 : x_2]
    #         Iyy_window = Iyy[y_1 : y_2, x_1 : x_2]
            
    #         det = Ixx_window.sum() * Iyy_window.sum() - Ixy_window.sum() ** 2
    #         tr = Ixx_window.sum() + Iyy_window.sum()
    #         R = det - k * (tr ** 2)
            
    #         if R > threshold:
    #             corner_x.append(x)
    #             corner_y.append(y)
                 
    # x = np.array(corner_x)
    # y = np.array(corner_y)
    # return x, y
    

def get_features(image, x, y, feature_width):
    """ Returns a set of feature descriptors for a given set of interest points. 
    Args:
        image - can be grayscale or color, your choice.
        x and y: nx1 vectors of x and y coordinates of interest points.
            The local features should be centered at x and y.
        feature_width - in pixels, is the local feature width. You can assume
            that feature_width will be a multiple of 4 (i.e. every cell of your
            local SIFT-like feature will have an integer width and height).
        If you want to detect and describe features at multiple scales or
            particular orientations you can add other input arguments.
    Returns:
        features: the array of computed features. It should have the
            following size: [length(x) x feature dimensionality] (e.g. 128 for
            standard SIFT)
    """
    # Initialization
    num_pts = x.shape[0]
    num_cell = 4
    h, w = image.shape[0], image.shape[1]
    win_bound = int(feature_width/num_cell)
    features = np.zeros((x.shape[0], 128))
    
    # Step 1: Gaussian Kernel
    img_gaussian = (cv2.GaussianBlur(image, (7,7), 1) * 255).astype(np.uint8)
    
    # Step 2: Image gradient
    dx, dy = cv2.spatialGradient(img_gaussian)
    dx, dy = dx.astype(np.float32), dy.astype(np.float32)
    
    # Step 3: Magnitude and orientation for each pixel
    mag = np.sqrt(dx**2+dy**2)
    orient = np.arctan2(dy, dx)
    orient[orient < 0] += 2*np.pi
    
    # Step 4: Splitting into 16 cells & Magnitude for each cell further weighted
    mag_gaussian = cv2.GaussianBlur(mag, (feature_width-1, feature_width-1), feature_width/2)
    
    for idx in range(num_pts):
        h_bound, w_bound = h - (1 + feature_width), w - (1 + feature_width)
        x_bound, y_bound = x[idx] - feature_width/2, y[idx] - feature_width/2
        x_start = int(max(0, min(x_bound, w_bound)))
        y_start = int(max(0, min(y_bound, h_bound)))
        
    # Step 5: Cast the orientations for each pixels into 8 bins
        for y_i in range(4):
            for x_i in range(4):
                bins = np.zeros(8)
                x_left, y_left = x_start + x_i * win_bound, y_start + y_i * win_bound
                x_right, y_right = x_left + win_bound, y_left + win_bound
                
                mag_window = mag_gaussian[y_left:y_right, x_left:x_right].reshape(-1)
                orient_window = orient[y_left:y_right, x_left:x_right].reshape(-1)

                for i in range(bins.shape[0]):
                    angles = np.all([orient_window >= (i*np.pi/4), orient_window < ((i+1)*np.pi/4)],0)
                    bins[i] += np.sum(mag_window[angles])
                features[idx, (x_i * 4 + y_i) * 8:(x_i * 4 + y_i) * 8 + 8] = bins
                
    # Step 6: Normalize the feature vector
        eps = 1e-5
        features[idx,:] /= np.sqrt(np.sum(features[idx,:]**2)) + eps
    
    return features

def match_features(features1, features2, threshold=0.0):
    """ 
    Args:
        features1 and features2: the n x feature dimensionality features
            from the two images.
        threshold: a threshold value to decide what is a good match. This value 
            needs to be tuned.
        If you want to include geometric verification in this stage, you can add
            the x and y locations of the features as additional inputs.
    Returns:
        matches: a k x 2 matrix, where k is the number of matches. The first
            column is an index in features1, the second column is an index
            in features2. 
        Confidences: a k x 1 matrix with a real valued confidence for every
            match.
        matches' and 'confidences' can be empty, e.g. 0x2 and 0x1.
    """
    # Initialization
    num_to_match = max(features1.shape[0], features2.shape[0])
    matched = np.zeros((num_to_match, 2))
    confidence = np.zeros((num_to_match,1))
    
    # Step 1: Compute the feature distances from every point in feature 1 to 2
    for i in range(features1.shape[0]):
        distance_list = np.sqrt(np.sum((features1[i] - features2)**2, axis=1))
        min_1 = np.min(distance_list)
        min_2 = np.sort(distance_list)[1]
    # Step 2: Compute the ratio and confidence of point i
        ratio = min_1 / min_2
        conf_temp = 1/ratio
    # Step 3: Find points with confidence larger than threshold
        if conf_temp > threshold:
            confidence[i] = conf_temp
            matched[i] = i, np.argmin(distance_list)
    max_idx = np.argsort(confidence, axis=0)[::-1, 0]
    confidence, matched = confidence[max_idx,:], matched[max_idx,:]

    return matched, confidence