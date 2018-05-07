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

    def __init__(self, filename,process_width=20):
        self.process_width = process_width
        self.grey_img = cv2.imread(filename, 0)
        if self.grey_img is None:
            raise FileNotFoundError
        self.col_img = cv2.imread(filename)
        self.bin_img = inv(cv2.adaptiveThreshold(
                inv(self.grey_img, 255), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 0))
        self.staff_black, self.staff_white = 0, 0
        self.staff_black, self.staff_white = self.calculate_staff_values()
        self.seeds = None
        self.staff_removed = None
        self.staff_lines = None


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
                             range(x, 0, int(-staff_white / 2)))
                track(y, staff_res, grey_img, staff_black,
                      range(x, staff_res.shape[1], int(staff_white / 2)), line=line)
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
        img = np.array(self.bin_img,copy=True)
        staff_black, staff_white = self.calculate_staff_values()
        staff_lines, _ = self.get_staff_lines_from_seeds()
        for line in staff_lines:
            line = sorted(line, key=itemgetter(0))
            (prev_x, prev_y) = line[0]
            for (x, y) in line:
                for xx in range(prev_x, x):
                    y_int = round(prev_y + (y - prev_y) * (xx - prev_x) / (x - prev_x))
                    if check_above_amount(img, xx, y - 1, math.ceil(staff_black / 2)) > 0 and \
                            check_below_amount(img, xx, y + staff_black, math.ceil(staff_black / 2)) > 0:
                        continue
                    increase_by = 0
                    above_amount = check_above_amount(img, xx, y - 1, math.ceil(staff_black / 2) + 1)
                    if above_amount != math.ceil(staff_black / 2) + 1:
                        y_int -= above_amount
                        increase_by += above_amount
                    below_amount = check_below_amount(img, xx, y + staff_black, math.ceil(staff_black / 2) + 1)
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