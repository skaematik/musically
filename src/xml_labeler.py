import os
import shutil

import cv2
from xml.dom import minidom
from Segmenter import Segmenter
import xml.etree.ElementTree as ET
from classifier import Classifier
from Song import Song


def is_barline(symbol):
    return symbol.h/symbol.w > 8


class Xml_labeler:

    def __init__(self, picture_filenames, xml_filename, output_path):
        self.picture_filenames = picture_filenames
        self.xmldoc = minidom.parse(xml_filename)
        self.tree = ET.parse(xml_filename)
        self.output_path = output_path
        self.matched_pairs = []

    def label_symbols(self, min_x_pos=500, edit_last=False, dont_use_cache=False):
        measures = self.xmldoc.getElementsByTagName('Measure')
        xml_objects = []
        while len(measures) is not 0:
            current_measure = measures.pop(0)
            current_measure_number = current_measure.getAttribute('number')
            for child in current_measure.childNodes:
                if child.localName is not None:
                    if child.nodeName == 'Clef':
                        xml_objects.append(('clef', current_measure_number))
                    elif child.nodeName == 'TimeSig':
                        xml_objects.append(('timeSig', current_measure_number))
                    elif child.nodeName == 'Rest':
                        xml_objects.append(('rest_{}'.format(
                            child.childNodes[1].childNodes[0].data), current_measure_number))
                    elif child.nodeName == 'Chord':
                        if child.childNodes[1].nodeName == 'BeamMode':
                            xml_objects.append(('tied_{}'.format(child.childNodes[3].childNodes[0].data),
                                                current_measure_number))
                        elif child.childNodes[3].nodeName != 'Beam':
                            xml_objects.append(('{}'.format(child.childNodes[1].childNodes[0].data),
                                                current_measure_number))
            xml_objects.append(('barline', current_measure_number))
        first_pic = True
        for filename in self.picture_filenames:
            segmenter = Segmenter.load_segmenter_from_file(filename)
            if (segmenter is None) or dont_use_cache:
                segmenter = Segmenter(filename)
                if not first_pic:
                    if edit_last or filename != self.picture_filenames[-1]:
                        segmenter.blockout_markings()
                segmenter.remove_staff_lines()
                segmenter.getSymbols(True)
                segmenter.save_to_file()
            symbols = segmenter.symbols
            i = -1
            saveCount = 0
            while len(symbols) != 0:
                i += 1
                saveCount = saveCount - 1 if saveCount > 0 else 0
                if not is_barline(symbols[0]) and xml_objects[0][0] == 'barline':
                    saveCount = 0
                    print('error not enough xmls found at bar {} line {} in file {}'.format(self.matched_pairs[-1][0][1],
                                                                                            symbols[0].line_num,
                                                                                            filename[-20:-4]))
                    popped = self.matched_pairs.pop()
                    while popped[0][0] != 'barline':
                        popped = self.matched_pairs.pop()
                    self.matched_pairs.append(popped)
                    popped = symbols.pop(0)
                    while not is_barline(popped):
                        popped = symbols.pop(0)
                    symbols.insert(0, popped)
                if is_barline(symbols[0]) and xml_objects[0][0] != 'barline':
                    saveCount = 0
                    print('error too many xmls found at bar {} line {} in file {}'.format(
                        self.matched_pairs[-1][0][1], symbols[0].line_num, filename[-20:-4]))
                    # save = True
                    popped = self.matched_pairs.pop()
                    while popped[0][0] != 'barline':
                        popped = self.matched_pairs.pop()
                    self.matched_pairs.append(popped)
                    popped = xml_objects.pop(0)
                    while popped[0] != 'barline':
                        popped = xml_objects.pop(0)
                    xml_objects.insert(0, popped)
                if symbols[0].x < min_x_pos and (not first_pic or i != 0):
                    self.matched_pairs.append((('clef', - 1), symbols.pop(0)))
                else:
                    self.matched_pairs.append(
                        (xml_objects.pop(0), symbols.pop(0)))
                if saveCount > 0:
                    thumbnail_name = os.path.join(self.output_path,
                                                  '{}_{}_line{}__{}.png'.format(filename[-13:-10], self.matched_pairs[-1][0][0],
                                                                                self.matched_pairs[-1][1].line_num, i))
                    cv2.imwrite(thumbnail_name, self.matched_pairs[-1][1].im)
            first_pic = False

    def save_symbols_pictures(self, use_folders=True, name_format='note_{}_{}.png'):
        classes = set()
        for m in self.matched_pairs:
            classes.add(m[0][0])  # in case we add more later
        if use_folders:
            for c in classes:
                try:
                    shutil.rmtree(os.path.join(self.output_path, c))
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
                os.mkdir(os.path.join(self.output_path, c))
            for i in range(len(self.matched_pairs)):
                cv2.imwrite(
                    os.path.join(
                        os.path.join(self.output_path,
                                     self.matched_pairs[i][0][0]),
                        name_format.format(self.matched_pairs[i][0][0], i)),
                    self.matched_pairs[i][1].im)
        else:
            for i in range(len(self.matched_pairs)):
                cv2.imwrite(
                    os.path.join(
                        self.output_path, name_format.format(self.matched_pairs[i][0], i)),
                    self.matched_pairs[i][1].im)

    def compare(self):
        pitch_mappings = {
            48: "C3", 50: "D3", 52: "E3", 53: "F3", 55: "G3", 57: "A3", 59: "B3",
            60: "C4", 62: "D4", 64: "E4", 65: "F4", 67: "G4", 69: "A4", 71: "B4",
            72: "C5", 74: "D5", 76: "E5", 77: "F5", 79: "G5", 81: "A5", 83: "B5",
            84: "C6", 86: "D6", 88: "E6", 89: "F6", 91: "G6", 93: "A6", 95: "B6",
        }

        measures = self.xmldoc.getElementsByTagName('Measure')
        xml_objects = []
        while len(measures) is not 0:
            current_measure = measures.pop(0)
            current_measure_number = current_measure.getAttribute('number')
            for child in current_measure.childNodes:
                if child.localName is not None:
                    if child.nodeName == 'Clef':
                        xml_objects.append(('clef', current_measure_number))
                    elif child.nodeName == 'TimeSig':
                        xml_objects.append(('timeSig', current_measure_number))
                    elif child.nodeName == 'Rest':
                        xml_objects.append(
                            ('rest_{}'.format(child.childNodes[1].childNodes[0].data), current_measure_number))
                    elif child.nodeName == 'Chord':
                        if child.childNodes[1].nodeName == 'BeamMode':
                            xml_objects.append(('{}'.format(child.childNodes[3].childNodes[0].data),
                                                current_measure_number,
                                                child.childNodes[7].childNodes[1].childNodes[0].data))
                        elif child.childNodes[3].nodeName == 'Beam':
                            xml_objects.append(('{}'.format(child.childNodes[1].childNodes[0].data),
                                                current_measure_number,
                                                child.childNodes[5].childNodes[1].childNodes[0].data))
                        elif child.childNodes[3].nodeName != 'Beam':
                            xml_objects.append(('{}'.format(child.childNodes[1].childNodes[0].data),
                                                current_measure_number,
                                                child.childNodes[3].childNodes[1].childNodes[0].data))
            xml_objects.append(('barline', current_measure_number))

        xml_symbols = []
        for m in xml_objects:
            if m[0] != "clef" and m[0] != "timeSig" and m[0] != "barline":
                if len(m) == 3:
                    xml_symbols.append((m[0], m[1], pitch_mappings[int(m[2])]))
                else:
                    xml_symbols.append(m)

        xml_bars = []
        curr_bar = []
        curr_bar_num = 1
        i = 0
        while i < len(xml_symbols):
            sym = xml_symbols[i]
            if int(sym[1]) > curr_bar_num:
                xml_bars.append(curr_bar)
                curr_bar = []
                curr_bar_num += 1
            curr_bar.append(sym)
            i += 1

        symbols = [m[1] for m in self.matched_pairs]
        classifier = Classifier()
        y = classifier.predict_symbols(symbols, use_class_numbers=True)
        for i in range(len(y)):
            symbols[i].work_out_type(y[i])

        song = Song()
        song.add_symbols(symbols)
        song.parse_symbols()
        stream = song.stream

        stream_bars = []
        curr_bar = []
        curr_bar_time = 0
        i = 0
        while i < len(stream):
            s = stream[i]
            sym_duration_num = s.duration.quarterLength
            curr_bar.append(s)
            curr_bar_time += sym_duration_num
            if curr_bar_time == 4:
                curr_bar_time = 0
                stream_bars.append(curr_bar)
                curr_bar = []
            i += 1

        def compare_bar(xmlbar, streambar, xmli, strmj):
            total_symbols = len(xmlbar)
            total_pitched = 0

            duration_errors = 0
            pitch_errors = 0

            both_correct = 0

            i = 0
            while True:
                if i == len(xmlbar) or i == len(streambar):
                    break
                xml_sym = xmlbar[i]
                stream_sym = streambar[i]

                if xml_sym[0].find("rest") != -1:
                    xml_duration = xml_sym[0][5:]
                    if xml_duration == "measure":
                        xml_duration = "whole"
                    xml_value = "rest"
                else:
                    xml_duration = xml_sym[0]
                    xml_value = xml_sym[2]
                    total_pitched += 1

                sym_duration = stream_sym.duration.type  # str
                sym_value = stream_sym.nameWithOctave if stream_sym.isNote else "rest"

                if xml_duration == sym_duration and xml_value == sym_value:
                    both_correct += 1

                if xml_duration != sym_duration:
                    duration_errors += 1
                if xml_value != "rest" and sym_value != "rest":
                    if xml_value != sym_value:
                        pitch_errors += 1
                i += 1
            return duration_errors, pitch_errors, both_correct, total_symbols, total_pitched

        i = 0
        j = 0
        incorrect_duration_count = 0
        incorrect_pitch_count = 0

        total_measures = min(len(xml_bars), len(stream_bars))
        both_correct_count = 0

        total_xml_symbols = 0
        total_xml_pitch_notes = 0

        while True:
            if i == len(xml_bars) or j == len(stream_bars):
                break
            bar_duration_err, bar_pitch_err, both_correct, bar_symbols, bar_pitched = compare_bar(
                xml_bars[i], stream_bars[j], i, j)
            if bar_duration_err > 1 and bar_pitch_err > 1:
                i += 1
                continue
            else:
                print("bar", i, j, "duration errors =",
                      bar_duration_err, "pitch errors =", bar_pitch_err)
                incorrect_duration_count += bar_duration_err
                incorrect_pitch_count += bar_pitch_err

                total_xml_symbols += bar_symbols
                total_xml_pitch_notes += bar_pitched

                both_correct_count += both_correct

                i += 1
                j += 1

        print("both correct =", both_correct_count, "/", total_xml_symbols,
              "=", both_correct_count / total_xml_symbols * 100, "%")
        print("pitch fail rate =", incorrect_pitch_count, "/", total_xml_pitch_notes, "=", incorrect_pitch_count / total_xml_pitch_notes * 100,
              "%")
        print("duration fail rate =", incorrect_duration_count, "/", total_xml_symbols, "=",
              incorrect_duration_count / total_xml_symbols * 100,
              "%")


def main():
    picture_filenames = []
    files = sorted(os.listdir('./resources/sheets/training/pieces/'))
    for filename in files:
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            print(filename)
            img = cv2.imread(
                './resources/sheets/training/pieces/{}.png'.format(filename[:-4]))
            top = int(0.1 * img.shape[0])  # shape[0] = rows
            img = cv2.copyMakeBorder(
                img, top, top, top, top, cv2.BORDER_CONSTANT, None, 255)
            cv2.imwrite('./resources/sheets/tmp/{}_noise.png'.format(filename[:-4]), img)
            picture_filenames.append(
                './resources/sheets/tmp/{}_noise.png'.format(filename[:-4]))
    xml_labeler = Xml_labeler(
        picture_filenames=picture_filenames,
        xml_filename=os.path.join(
            './resources/sheets/training/pieces/', 'auto_gen_large.mscx'),
        output_path='./tests/output/')
    xml_labeler.label_symbols(edit_last=True)
    # xml_labeler.save_symbols_pictures()
    xml_labeler.compare()


if __name__ == "__main__":
    main()
