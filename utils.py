from collections import Counter
import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def import_img_greyscale(file_name):
    """Loads image from file as greyscale
        :raises FileNotFoundError
        :returns greyscale image"""
    img = cv2.imread(file_name, 0)
    if img is None:
        raise FileNotFoundError
    return img


def inv(img, val=255):
    """ Inverts image
        :arg img
            image to invert
        :arg val
            maximum value in image
        :returns inverted image"""
    return 255-img


def find_staffs(img, show_plots=False, top_buffer=20, left_buffer=20, min_width=0.8, display_for=10000):
    """Finds each line of music in the img and returns each in a list
    Steps:
    1) dialate the image by same width as staff height to ensure connectedness
    2) apply connected component algorithm
    3) segment into separate images any labeled image that is at least min_width
    fraction of total width long.

    If two stave lines are connected into a stanza they will return as one object.
    Currently any other things inside the bounding box are also returned, but we can trim them
    if needed.
    :param img: binary image, staff needs to be (roughly) horizontal, make sure background is white
    :param show_plots: where or not to show the results, will pause until keypress
    :param display_for: length of time to display plots for
    :return: a list of np.array for each staff detected. will attempt to center each one with
     top_buffer and left_buffer whitespace.
    """
    staff_width, staff_height = calculate_staff_values(img)
    out_erroded = cv2.erode(img, np.ones((staff_height, staff_height)))
    if show_plots:
        cv2.imshow('image segmentation dialated',out_erroded)
    labeled_image, num_features = ndimage.label(inv(out_erroded))
    # Find the location of all objects
    objs = ndimage.find_objects(labeled_image)
    # Get the height and width
    imgs = []
    for ob in objs:
        if (ob[1].stop - ob[1].start) > min_width*img.shape[1]:
            imgs.append(img[ob])
            if show_plots:
                cv2.imshow('image segment {}'.format(len(imgs)), imgs[-1])
    if show_plots:
        plt.imshow(labeled_image)
        plt.show()
        cv2.waitKey(display_for)
    return imgs


def calculate_staff_values(img):
    """Calculates the staff_width and staff height
        staffwidth is the thickness of the black lines
        staffheight is the white between black lines
        most common runtime encodingnto calculate values
        :arg img
            binary image of music, makr sure music is in horizontal lines

        :returns staff_width, staff_height"""
    blackruns = Counter()
    whiteruns = Counter()  # key is run length val is count of this run length
    is_white = True
    for x in range(img.shape[1]):
        count = 0
        for y in range(img.shape[0]):
            cur_white = img[y, x] == 255
            if cur_white != is_white:
                if is_white:
                    whiteruns[count] += 1 # record this run
                else:
                    blackruns[count] += 1  # record this run
                count = 0
            is_white = cur_white
            count += 1
    print('blackRun: {}', blackruns.most_common(1))
    print('whiteRun: {}', whiteruns.most_common(1))
    return blackruns.most_common(1)[0][0], whiteruns.most_common(1)[0][0]


def boundingboxes(im, merge_overlap=False):
    """draws bounding boxes around detected objects
    
    im: black and white image - black background, white foreground
    merge_overlap: whether to merge overlapping bounding boxes
    
    returns: list of boxes in tuple (x,y,width,height) format, rgb image with boxes drawn
    """
    def draw(im, boxes):
        for x, y, w, h in boxes:
            cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), thickness=1)
    
    def box2pts(box):
        x, y, w, h = box
        return (x, y), (x + w, y + h)
    
    def merge_overlaps(boxes):
        def overlapping(b1, b2):
            l1, r1 = box2pts(b1)
            l2, r2 = box2pts(b2)
            return not (l1[0] > r2[0] or l2[0] > r1[0] or l1[1] > r2[1] or l2[1] > r1[1])
        
        def next():
            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i < j and overlapping(boxes[i], boxes[j]):
                        b1 = boxes.pop(j)
                        b2 = boxes.pop(i)
                        return b1, b2
            return None
        
        while True:
            pair = next()
            if pair is None:
                break
            b1, b2 = pair
            l1, r1 = box2pts(b1)
            l2, r2 = box2pts(b2)
            
            x = min(l1[0], l2[0])
            x_ = max(r1[0], r2[0])
            y = min(l1[1], l2[1])
            y_ = max(r1[1], r2[1])
            w, h = x_ - x, y_ - y
            boxes.append((x, y, w, h))
        
        
    _, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    boxes = [cv2.boundingRect(c) for c in contours]
    
    colour = np.dstack((im, im, im))
    
    if merge_overlap:
        merge_overlaps(boxes)
    
    draw(colour, boxes);
    
    return boxes, colour
