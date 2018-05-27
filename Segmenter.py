import os
from operator import itemgetter
import math
from collections import Counter
from operator import itemgetter

import cv2

from utils import *


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
            self.grey_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        self.bin_img = inv(self.bin_img)
        self.staff_black, self.staff_white = 0, 0
        self.staff_black, self.staff_white = self.calculate_staff_values()
        self.seeds = None
        self.staff_removed = None
        self.staff_lines = None
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
        self.staff_lines = staff_lines
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
        im = cv2.dilate(im, np.ones((5,5)))
        def boxes2symbols(boxes):
            symbols = []
            for x, y, w, h in boxes:
                line_num = self.get_line_number(x, y, w, h)
                if line_num != -1:
                    symbols.append(Symbol(SymbolType.UNKNOWN, x, y, w, h, line_num))
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

        _, contours, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        new_c = []
        for c in contours:
            if cv2.contourArea(c) > 100:
                new_c.append(c)
        boxes = [cv2.boundingRect(c) for c in new_c]

        colour = np.dstack((im, im, im))

        if merge_overlap:
            merge_overlaps(boxes)

        draw(colour, boxes);

        symbols = boxes2symbols(boxes)
        line_len = self.bin_img.shape[1]
        symbols = sorted(symbols, key=lambda sym: sym.line_num*line_len+sym.x)
        self.symbols = symbols
        self.add_in_pictures()
        return colour

    def get_line_number(self, x,y,w,h):
        """ get closest line for a y co-ordinate
        x needed as staff lines may not be straight, -1 if not on a line, lines start from 0"""
        staff_black, staff_white = self.calculate_staff_values()
        staff_lines, _ = self.get_staff_lines_from_seeds()
        lines = []
        current_line = -1
        prev_y_val = 0
        for line in staff_lines:
            lines.append(sorted(line, key=itemgetter(0)))
        lines = sorted(lines, key=lambda line: line[len(line)//2][1])
        for line in lines:
            if (line[len(line)//2][1] - prev_y_val) > staff_white *2:
                current_line += 1
            prev_y_val = line[len(line) // 2][1]
            if y-staff_white < prev_y_val < (y + h+staff_white):
                return current_line
            if prev_y_val > y+staff_white*20: # give up
                return -1
        return -1


    def add_in_pictures(self, from_staff_removed=True, fixed_width=True, width=150):
        """
        adds images into symbol objects. none fixed width does not work yet
        :param from_staff_removed:
        :param fixed_width:
        :param width:
        :return:
        """
        base_img = self.staff_removed if from_staff_removed else self.grey_img
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
            sym.im = im

    def saveSymbols(self, format, save_origonal=False, path='./', width=150,reject_ratio=100,min_area=0,reject_path='./', dirty_times=0):
        base_img = self.grey_img if save_origonal else self.staff_removed
        count=0
        o_width = width
        for sym in self.symbols:
            scale = 1

            while max(max(sym.w, sym.h), width) != width:
                width *= 2
                scale *= 2  # use later
            offsetx = (width - sym.w) // 2
            offsety = (width - sym.h) // 2
            im = np.ones((width,width),dtype=np.uint8)*255  # white
            im[(offsety):(sym.h + offsety), (offsetx):(sym.w + offsetx)] = \
                base_img[sym.y:(sym.y+sym.h), sym.x:(sym.x+sym.w)]
            im = cv2.resize(im, (o_width, o_width), interpolation=cv2.INTER_CUBIC)

            if sym.h/sym.w > reject_ratio:
                cv2.imwrite(os.path.join(reject_path, 'ratio_'+format.format(count)), im)
            elif sym.w * sym.h < min_area:
                cv2.imwrite(os.path.join(reject_path, 'size_'+format.format(count)), im)
            else:
                if dirty_times == 0:
                    cv2.imwrite(os.path.join(path,format.format(count)), im)
                else:
                    for i in range(dirty_times):
                        cv2.imwrite(os.path.join(path, format.format(count)), dirty(im))
                        count += 1
            count += 1

    def blockout_markings(self):
        # page_num
        for y in range(3600,3700):
            for x in range (2500,2800):
                self.grey_img[y,x] = 255
                self.bin_img[y, x] = 255
        for y in range(3550, 3750):
            for x in range(300, 800):
                self.grey_img[y, x] = 255
                self.bin_img[y, x] = 255
        line_nums = [185, 465, 740, 1020, 1300, 1575, 1855, 2135, 2410, 2690, 2968]
        for line in line_nums:
            line+=350
            for y in range(line, line+35):
                for x in range(400, 515):
                    self.grey_img[y, x] = 255
                    self.bin_img[y, x] = 255



    def save_to_file(self):

