import os
import shutil

import cv2
from xml.dom import minidom
from Segmenter import Segmenter
import xml.etree.ElementTree as ET


def is_barline(symbol):
    return symbol.h/symbol.w > 8


class Xml_labeler:

    def __init__(self, picture_filenames, xml_filename, output_path):
        self.picture_filenames = picture_filenames
        self.xmldoc = minidom.parse(xml_filename)
        self.tree = ET.parse(xml_filename)
        self.output_path = output_path
        self.matched_pairs = []

    def label_symbols(self, min_x_pos=500, edit_last=False):
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
                        xml_objects.append(('rest_{}'.format(child.childNodes[1].childNodes[0].data),current_measure_number))
                    elif child.nodeName == 'Chord':
                        if child.childNodes[1].nodeName == 'BeamMode':
                            xml_objects.append(('tied_{}'.format(child.childNodes[3].childNodes[0].data), current_measure_number))
                        elif child.childNodes[3].nodeName != 'Beam':
                            xml_objects.append(('{}'.format(child.childNodes[1].childNodes[0].data), current_measure_number))
            xml_objects.append(('barline', current_measure_number))
        first_pic = True
        for filename in self.picture_filenames:
            segmenter = Segmenter(filename)
            if not first_pic:
                if edit_last or filename != self.picture_filenames[-1]:
                    segmenter.blockout_markings()
            segmenter.remove_staff_lines()
            segmenter.getSymbols(True)
            symbols = segmenter.symbols
            # print("\n".join(xml_objects))
            save = False
            i = -1
            saveCount = 0
            while len(symbols) != 0:
                i += 1
                saveCount = saveCount - 1 if saveCount > 0 else 0
                if not is_barline(symbols[0]) and xml_objects[0][0] == 'barline':
                    saveCount = 10
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
                    saveCount = 10
                    print('error too many xmls found at bar {} line {} in file {}'.format(self.matched_pairs[-1][0][1],symbols[0].line_num, filename[-20:-4]))
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
                    self.matched_pairs.append((('clef',- 1), symbols.pop(0)))
                else:
                    self.matched_pairs.append((xml_objects.pop(0), symbols.pop(0)))
                if saveCount > 0:
                    thumbnail_name = os.path.join(self.output_path,
                                 '{}_{}_line{}__{}.png'.format(filename[-13:-10], self.matched_pairs[-1][0][0],
                                                               self.matched_pairs[-1][1].line_num, i))
                    cv2.imwrite(thumbnail_name, self.matched_pairs[-1][1].im)
            first_pic = False



    def save_symbols_pictures(self, use_folders=True, name_format='note_{}_{}.png'):
        classes = set()
        for m in self.matched_pairs:
            classes.add(m[0][0]) # in case we add more later
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
                        os.path.join(self.output_path, self.matched_pairs[i][0][0]),
                        name_format.format(self.matched_pairs[i][0][0], i)),
                    self.matched_pairs[i][1].im)
        else:
            for i in range(len(self.matched_pairs)):
                cv2.imwrite(
                    os.path.join(
                        self.output_path, name_format.format(self.matched_pairs[i][0], i)),
                    self.matched_pairs[i][1].im)
