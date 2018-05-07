import cv2
import numpy as np
def inv(img, val=255):
    """ Inverts image
        :arg img
            image to invert
        :arg val
            maximum value in image
        :returns inverted image"""
    return 255-img


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
