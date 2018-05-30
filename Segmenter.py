import os
import pickle
import shutil
from operator import itemgetter
import math
from collections import Counter
from operator import itemgetter
import scipy

import cv2

from utils import *

cache_path = './cached_segmenter'


class Segmenter:
    """
    Class that does everything about inital processing of the image
    holds:
    grey_img: greyscale image
    bin_img: binary thresholded image
    col_img: colour image
    staff_removed: image with staff removed
    staffs: list of lists (x,y) of points along the staff line
    boxes: tuples of (x,y,width,height) where ROIs are
    staff_black: thickness of black lines
    staff_white: thickness of white lines
    process_wdith: used in staff detection and seeding, will search the middle+- width/process_wdith, default will
    search 10%.
    DONT EDIT THESE IMAGES MAKE A COPY INSTEAD
    """

    def __init__(self, filename, process_width=20):
        self.process_width = process_width
        self.grey_img = cv2.imread(filename, 0)
        if self.grey_img is None:
            raise FileNotFoundError
        self.col_img = cv2.imread(filename)
        _, self.bin_img = cv2.threshold(
            self.grey_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.bin_img = inv(self.bin_img)
        self.staff_black, self.staff_white = 0, 0
        self.staff_black, self.staff_white = self.calculate_staff_values()
        self.seeds = None
        self.staff_removed = None
        self.staff_lines = None
        self.grouped_staff_lines = None
        self.symbols = None
        self.filename = filename

    def calculate_staff_values(self):
        """Calculates the staff_width and staff height
            staffwidth is the thickness of the black lines
            staffheight is the white between black lines
            most common runtime encoding to calculate values
            :returns staff_black, staff_white"""
        if self.staff_black != 0:
            return self.staff_black, self.staff_white  # cached
        black_runs = Counter()
        white_runs = Counter()  # key is run length val is count of this run length
        is_white = True
        for x in range(int(self.bin_img.shape[1] / 2 - self.bin_img.shape[1] / self.process_width),
                       int(self.bin_img.shape[1] / 2 + self.bin_img.shape[1] / self.process_width)):
            count = 0
            for y in range(self.bin_img.shape[0]):
                cur_white = self.bin_img[y, x] == 255
                if cur_white != is_white:
                    if is_white:
                        white_runs[count] += 1  # record this run
                    else:
                        black_runs[count] += 1  # record this run
                    count = 0
                is_white = cur_white
                count += 1
        return black_runs.most_common(1)[0][0], white_runs.most_common(1)[0][0]

    def get_staff_seeds(self, black_margin=1, white_margin=2, debug=False):
        """ just call get_staffs
        :return: an array of tuples (x values, list of y values) corresponding
            to where the black lines start
        """
        # state 0 white before staff
        # state 1 black of 1st line
        # state 2 white of 1st space
        # state 3 black of 2nd line
        # state 4 white of 2nd space
        # state 5 black of 3rd line
        # state 6 white of 3rd space
        # state 7 black of 4th line
        # state 8 white of 4th space
        # state 9 black of 5th line
        # state 10 white after staff
        # state 11 staff found
        # even is white states
        # odd is black states
        if self.seeds is not None and not debug:
            return self.seeds, None  # cached
        staff_black, staff_white = self.calculate_staff_values()
        colour_img = None
        if debug:
            colour_img = np.array(self.col_img, copy=True)
        seeds = []
        current_seed = []
        img = self.bin_img
        for x in range(int(img.shape[1] / 2 - img.shape[1] / 10), int(img.shape[1] / 2 + img.shape[1] / 10)):
            state = 0
            count = 0
            for y in range(img.shape[0]):
                cur_white = img[y, x] == 255
                if state == 0:
                    if not cur_white:
                        if count > staff_white:
                            count = 1
                            state = 1  # must be black of top line
                            current_seed.append(y)
                    else:
                        count += 1
                elif state == 10:
                    if not cur_white:
                        count = 0
                        state = 0  # so close but its all over now
                        current_seed = []
                    else:
                        count += 1
                        if count > staff_white + white_margin:
                            seeds.append((x, current_seed))
                            current_seed = []
                            # done
                            state = 0
                            count = 0
                elif state % 2 == 1:  # BLACK LINE
                    if not cur_white:
                        if count < staff_black + black_margin:
                            count += 1
                        else:
                            state = 0
                            count = 0  # black is too long go back to first state
                            current_seed = []
                    else:
                        if abs(count - staff_black) <= black_margin:
                            count = 1
                            state += 1  # move onto white section
                        else:
                            state = 0
                            count = 0  # black is too short go back to first state
                            current_seed = []
                elif state % 2 == 0:  # WHITE LINE
                    if cur_white:
                        if count < staff_white + white_margin:
                            count += 1
                        else:
                            state = 0
                            count = 0  # white is too long go back to first state
                            current_seed = []
                    else:
                        if abs(count - staff_white) <= white_margin:
                            count = 1
                            state += 1  # move onto black section
                            current_seed.append(y)
                        else:
                            state = 0
                            count = 0  # white is too short go back to first state
                            current_seed = []
        if debug:
            for (x, ys) in seeds:
                for y in ys:
                    for yy in range(y, y + staff_black):
                        colour_img[yy, x] = np.array([0, 0, 255])
        self.seeds = seeds
        return seeds, colour_img

    def get_staff_lines_from_seeds(self, debug=False):
        """ Call get_staff_seeds if you want to modify params
        place markers along the staff lines, at spaces of staff_white/2
        data is not sorted.
        :return: array of arrays of tuples (x,y) corresponding to top of a staff line, not sorted
        """

        def track(y, staff_res, grey, staff_black, range_obj, line=None):
            if line is None:
                line = []
            for x in range_obj:
                sumLarge = np.sum(grey[y - staff_black:y + 2 * staff_black, x])
                sum = np.sum(grey[y:y + staff_black, x])
                if sumLarge < sum or (sumLarge - sum) < 255 * staff_black:
                    continue
                move = 0
                lowest = sum
                if np.sum(grey[y - 1:y - 1 + staff_black, x]) < sum:
                    move = -1
                    lowest = np.sum(grey[y - 1:y - 1 + staff_black, x])
                    if np.sum(grey[y - 2:y - 2 + staff_black, x]) < lowest:
                        move = -2
                        lowest = np.sum(grey[y - 2:y - 2 + staff_black, x])
                if np.sum(grey[y + 1:y + 1 + staff_black, x]) < lowest:
                    move = 1
                    lowest = np.sum(grey[y + 1:y + 1 + staff_black, x])
                    if np.sum(grey[y + 2:y + 2 + staff_black, x]) < lowest:
                        move = 2
                        np.sum(grey[y + 2:y + 2 + staff_black, x])
                # fill in

                y += move
                line.append((x, y))
                staff_res[y:y + staff_black, x] = np.ones((1, staff_black))
            return line

        if self.staff_lines is not None and not debug:
            return self.staff_lines, None
        colour_img = None
        if debug:
            colour_img = np.array(self.col_img, copy=True)
        grey_img = self.bin_img
        staff_black, staff_white = self.calculate_staff_values()
        seeds, _ = self.get_staff_seeds()
        done = []
        staff_res = 255 * np.ones(grey_img.shape).astype(np.uint8)
        staff_lines = []
        for (x, ys) in seeds:
            for y in ys:
                breakout = False
                for d in done:
                    if abs(d - y) < staff_black:
                        breakout = True
                        break
                if breakout:
                    continue
                line = track(y, staff_res, grey_img, staff_black,
                             range(x, 0, int(-staff_white / 4)))
                track(y, staff_res, grey_img, staff_black,
                      range(x, staff_res.shape[1], int(staff_white / 4)), line=line)
                staff_lines.append(line)
                done.append(y)
        if debug:
            for line in staff_lines:
                line = sorted(line, key=itemgetter(0))
                (prev_x, prev_y) = line[0]
                for (x, y) in line:
                    cv2.line(
                        colour_img,
                        (x, int(y + staff_black / 2)),
                        (prev_x, int(prev_y + staff_black / 2)),
                        (255, 0, 255), 2)
                    for yy in range(prev_y, prev_y + staff_black):
                        colour_img[yy, prev_x] = np.array([255, 0, 255])
                    prev_x, prev_y = x, y
        sorted_intra_lines = []
        for line in staff_lines:
            sorted_intra_lines.append(sorted(line, key=itemgetter(0)))
        sorted_lines = sorted(sorted_intra_lines, key=lambda line: line[len(line) // 2][1])
        grouped_staff_lines = []
        current_line = 0
        current_staff = []
        prev_y_val = 0
        for line in sorted_lines:
            if (line[len(line) // 2][1] - prev_y_val) > staff_white * 2:
                current_line += 1
                grouped_staff_lines.append(current_staff)
                current_staff = []
            prev_y_val = line[len(line) // 2][1]
            current_staff.append(line)
        grouped_staff_lines.append(current_staff) # append last line
        grouped_staff_lines.pop(0) # pop empty line -1
        self.staff_lines = staff_lines
        self.grouped_staff_lines = grouped_staff_lines
        return staff_lines, colour_img

    def remove_staff_lines(self):
        """ call earlier functions if you want to modify params"""

        def check_above_amount(img, x, y, limit):
            count = 0
            for yy in range(y, y - limit, -1):
                if img[yy, x] != 0:
                    return count
                count += 1
            return count

        def check_below_amount(img, x, y, limit):
            count = 0
            for yy in range(y, y + limit, 1):
                if img[yy, x] != 0:
                    return count
                count += 1
            return count

        if self.staff_removed is not None:
            return self.staff_removed
        img = np.array(self.bin_img, copy=True)
        staff_black, staff_white = self.calculate_staff_values()
        staff_lines, _ = self.get_staff_lines_from_seeds()
        for line in staff_lines:
            line = sorted(line, key=itemgetter(0))
            (prev_x, prev_y) = line[0]
            for (x, y) in line:
                for xx in range(prev_x, x):
                    y_int = round(prev_y + (y - prev_y) *
                                  (xx - prev_x) / (x - prev_x))
                    if check_above_amount(img, xx, y - 1, math.ceil(staff_black / 2)) > 0 and \
                            check_below_amount(img, xx, y + staff_black, int(math.ceil(staff_black / 2))) > 0:
                        continue
                    increase_by = 0
                    above_amount = check_above_amount(
                        img, xx, y - 1, int(math.ceil(staff_black / 2)) + 1)
                    if above_amount != math.ceil(staff_black / 2) + 1:
                        y_int -= above_amount
                        increase_by += above_amount
                    below_amount = check_below_amount(
                        img, xx, y + staff_black, int(math.ceil(staff_black / 2)) + 1)
                    if below_amount != math.ceil(staff_black / 2) + 1:
                        increase_by += below_amount
                    for yy in range(y_int, y_int + staff_black + increase_by):
                        img[yy, xx] = 255
                prev_x, prev_y = x, y
        img = inv(cv2.morphologyEx(
            inv(img), cv2.MORPH_CLOSE,
            np.ones((int(math.ceil(staff_black / 2)), int(math.ceil(staff_black / 2))), np.uint8)))
        img = inv(cv2.morphologyEx(
            inv(img), cv2.MORPH_OPEN,
            np.ones((int(math.ceil(staff_black / 2)), int(math.ceil(staff_black / 2))), np.uint8)))
        self.staff_removed = img
        return img

    def getSymbols(self, merge_overlap=False):
        """finds and draws bounding boxes around detected objects, returns boxes as Symbols


        merge_overlap: whether to merge overlapping bounding boxes

        returns: saves list of Symbols, returns rgb image with bounding boxes drawn
        """
        im = inv(self.staff_removed)
        im = cv2.dilate(im, np.ones((5, 5)))

        def boxes2symbols(boxes):
            symbols = []
            for x, y, w, h in boxes:
                line_num = self.get_line_number(x, y, w, h)
                if line_num != -1:
                    symbols.append(Symbol(SymbolType.UNKNOWN, x, y, w, h, line_num,self.grouped_staff_lines[line_num]))
            return symbols

        def draw(im, boxes):
            for x, y, w, h in boxes:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)

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
        
        
        def conncomp(im):
            def min_pix_dist(labels, label1, label2):
                idx1 = np.transpose(np.where(labels == label1))
                idx2 = np.transpose(np.where(labels == label2))
                return np.min(scipy.spatial.distance.cdist(idx1, idx2))
            
            def merge_boxes(b1, b2):
                l1, r1 = box2pts(b1)
                l2, r2 = box2pts(b2)

                x = min(l1[0], l2[0])
                x_ = max(r1[0], r2[0])
                y = min(l1[1], l2[1])
                y_ = max(r1[1], r2[1])
                w, h = x_ - x, y_ - y
                return x, y, w, h
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
            
            erased = {}
            
            for i in range(1, num_labels):
                if stats[i,cv2.CC_STAT_AREA] < 10*self.staff_white or stats[i,cv2.CC_STAT_WIDTH] > 20*self.staff_white or stats[i,cv2.CC_STAT_HEIGHT] > 20*self.staff_white:
                    labels[labels == i] = 0
                    erased[i] = True
            
            #stats = [label,col] -> col = cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA
            #standard crotchet is 80 by 30
            for i in range(1, num_labels):
                for j in range(1, num_labels):
                    if i < j and i not in erased and j not in erased:
                        centroid_i, centroid_j = centroids[i,:], centroids[j,:]
                        area_i, area_j = stats[i,cv2.CC_STAT_AREA], stats[j,cv2.CC_STAT_AREA]
                        x_i, x_j = stats[i,cv2.CC_STAT_LEFT], stats[j,cv2.CC_STAT_LEFT]
                        y_i, y_j = stats[i,cv2.CC_STAT_TOP], stats[j,cv2.CC_STAT_TOP]
                        w_i, w_j = stats[i,cv2.CC_STAT_WIDTH], stats[j,cv2.CC_STAT_WIDTH]
                        h_i, h_j = stats[i,cv2.CC_STAT_HEIGHT], stats[j,cv2.CC_STAT_HEIGHT]
                        
                        centroid_dist = np.linalg.norm(centroid_i - centroid_j)
                        """
                        bb_occupy_i, bb_occupy_j = area_i/(w_i*h_i), area_j/(w_j*h_j)
                        bb_occupy_varied = (bb_occupy_i < 0.5 and bb_occupy_j > 0.8) or (bb_occupy_j < 0.5 and bb_occupy_i > 0.8)
                        bb_occupy_low = (bb_occupy_i < 0.5 and bb_occupy_j < 0.5)
                        bb_occupy_high = (bb_occupy_i > 0.5 and bb_occupy_j > 0.5)
                        area_diff = abs(area_i - area_j)
                        area_perc_diff = 0.5*(float(area_diff/area_i) + float(area_diff/area_j))
                        """
                        
                        # use centroid comparison first cos pixel distances is slow
                        if centroid_dist < 30:
                            min_dist = min_pix_dist(labels, i, j)
                            # pretty close
                            if min_dist < 15:
                                # both fairly short -> whole notes split up
                                if h_i < 40 and h_j < 40:
                                    labels[labels == j] = i
                                    erased[j] = True
                                    stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT] = merge_boxes((x_i,y_i,w_i,h_i), (x_j,y_j,w_j,h_j))
            
            
            colour = np.zeros(im.shape + (3,), dtype="uint8")
            for label in range(1, num_labels):
                if label in erased:
                    continue
                idx = np.where(labels == label)
                for chan in range(3):
                    channel = colour[:,:,chan]
                    channel[idx] = np.random.randint(128, 256, dtype="uint8")
            return (num_labels, labels, erased, stats), colour
        
        """
        _, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_c = []
        for c in contours:
            if cv2.contourArea(c) > 100:
                new_c.append(c)
        
        boxes = [cv2.boundingRect(c) for c in new_c]

        #colour = np.dstack((im, im, im))
        
        if merge_overlap:
            merge_overlaps(boxes)

        #draw(colour, boxes);

        symbols = boxes2symbols(boxes)
        """
        
        conncomps, colour = conncomp(im)
        
        #line_len = self.bin_img.shape[1]
        #symbols = sorted(symbols, key=lambda sym: sym.line_num * line_len + sym.x)
        #self.symbols = symbols
        self.add_in_pictures(conncomps)
        return colour

    def get_line_number(self, x, y, w, h):
        """ get closest line for a y co-ordinate
        x needed as staff lines may not be straight, -1 if not on a line, lines start from 0"""
        staff_black, staff_white = self.calculate_staff_values()
        staff_lines, _ = self.get_staff_lines_from_seeds()
        lines = []
        current_line = -1
        prev_y_val = 0
        for line in staff_lines:
            lines.append(sorted(line, key=itemgetter(0)))
        lines = sorted(lines, key=lambda line: line[len(line) // 2][1])
        for line in lines:
            if (line[len(line) // 2][1] - prev_y_val) > staff_white * 2:
                current_line += 1
            prev_y_val = line[len(line) // 2][1]
            if y - staff_white < prev_y_val < (y + h + staff_white):
                return current_line
            if prev_y_val > y + staff_white * 20:  # give up
                return -1
        return -1

    def add_in_pictures(self, conncomps, from_staff_removed=True, fixed_width=True, width=300, final_width=150):
        """
        adds images into symbol objects. none fixed width does not work yet
        :param from_staff_removed:
        :param fixed_width:
        :param width:
        :return:
        """
        base_img = self.staff_removed if from_staff_removed else self.grey_img
        o_width = width
        
        symbols = []
        num_labels, labels, erased, stats = conncomps
        for i in range(1, num_labels):
            if i in erased:
                continue
            
            x,y,w,h = stats[i,cv2.CC_STAT_LEFT], stats[i,cv2.CC_STAT_TOP], stats[i,cv2.CC_STAT_WIDTH], stats[i,cv2.CC_STAT_HEIGHT]
            line_num = self.get_line_number(x, y, w, h)
            if line_num == -1:
                continue
            
            sym = Symbol(SymbolType.UNKNOWN, x, y, w, h, line_num,self.grouped_staff_lines[line_num])
            
            scale = 1
            width = o_width
            while max(max(sym.w, sym.h), width) != width:
                width *= 2
                scale *= 2  # use later
            offsetx = (width - sym.w) // 2
            offsety = (width - sym.h) // 2
            
            ref_im = np.ones_like(base_img) * 255
            pix_idx = np.where(labels == i)
            ref_im[pix_idx] = base_img[pix_idx]
            
            im = np.ones((width, width), dtype=np.uint8) * 255  # white
            im[(offsety):(sym.h + offsety), (offsetx):(sym.w + offsetx)] = \
                ref_im[sym.y:(sym.y + sym.h), sym.x:(sym.x + sym.w)]
            im = cv2.resize(im, (final_width, final_width), interpolation=cv2.INTER_CUBIC)
            sym.im = im
            sym.offsetx = offsetx
            sym.offsety = offsety
            sym.scale = scale
            sym.staff_white = self.staff_white
            sym.staff_black = self.staff_black
            
            symbols.append(sym)
        
        line_len = self.bin_img.shape[1]
        symbols = sorted(symbols, key=lambda sym: sym.line_num * line_len + sym.x)
        self.symbols = symbols
        
        """
        for i, sym in enumerate(symbols):
            cv2.imshow(str(i), sym.im)
        cv2.waitKey(0)
        """



    def saveSymbols(self, format, save_origonal=False, path='./', width=150, reject_ratio=100, min_area=0,
                    reject_path='./', dirty_times=0):
        base_img = self.grey_img if save_origonal else self.staff_removed
        count = 0
        o_width = width
        for sym in self.symbols:
            scale = 1
            while max(max(sym.w, sym.h), width) != width:
                width *= 2
                scale *= 2  # use later
            offsetx = (width - sym.w) // 2
            offsety = (width - sym.h) // 2
            im = np.ones((width, width), dtype=np.uint8) * 255  # white
            im[(offsety):(sym.h + offsety), (offsetx):(sym.w + offsetx)] = \
                base_img[sym.y:(sym.y + sym.h), sym.x:(sym.x + sym.w)]
            im = cv2.resize(im, (o_width, o_width), interpolation=cv2.INTER_CUBIC)
            if sym.h / sym.w > reject_ratio:
                cv2.imwrite(os.path.join(reject_path, 'ratio_' + format.format(count)), im)
            elif sym.w * sym.h < min_area:
                cv2.imwrite(os.path.join(reject_path, 'size_' + format.format(count)), im)
            else:
                if dirty_times == 0:
                    cv2.imwrite(os.path.join(path, format.format(count)), im)
                else:
                    for i in range(dirty_times):
                        cv2.imwrite(os.path.join(path, format.format(count)), dirty(im))
                        count += 1
            count += 1

    def blockout_markings(self):
        # page_num
        for y in range(3600, 3700):
            for x in range(2500, 2800):
                self.grey_img[y, x] = 255
                self.bin_img[y, x] = 255
        for y in range(3550, 3750):
            for x in range(300, 800):
                self.grey_img[y, x] = 255
                self.bin_img[y, x] = 255
        line_nums = [185, 465, 740, 1020, 1300, 1575, 1855, 2135, 2410, 2690, 2968]
        for line in line_nums:
            line += 350
            for y in range(line, line + 35):
                for x in range(400, 515):
                    self.grey_img[y, x] = 255
                    self.bin_img[y, x] = 255

    def save_to_file(self):
        pfilename = os.path.basename(self.filename)
        pfilename = pfilename[:-4]
        pfilename += '.pkl'
        try:
            os.mkdir(cache_path)
        except FileExistsError:
            pass
        with open(os.path.join(cache_path, pfilename), "wb") as pfile:
            pickle.dump(self, pfile, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_segmenter_from_file(filename):
        pfilename = os.path.basename(filename)
        pfilename = pfilename[:-4]
        pfilename += '.pkl'
        path = os.path.join(cache_path, pfilename)
        try:
            with open(path, "rb") as f:
                dump = pickle.load(f)
                return dump
        except FileNotFoundError:
            return None

    @staticmethod
    def symbols_from_File(filename, use_cache=True):
        segmenter = None
        if use_cache:
            segmenter = Segmenter.load_segmenter_from_file(filename)
        if segmenter is not None:
            #coloured = segmenter.getSymbols(True)
            #cv2.imwrite("symbols.png", coloured)
            return segmenter.symbols, segmenter
        segmenter = Segmenter(filename)
        segmenter.remove_staff_lines()
        coloured = segmenter.getSymbols(True)
        #cv2.imwrite("symbols.png", coloured)
        segmenter.save_to_file()
        return segmenter.symbols, segmenter
