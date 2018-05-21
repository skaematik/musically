import cv2
import os

from Segmenter import Segmenter
from utils import getSymbols


def main():
    sheet_path = './sheets/training/'
    output_path = './sheets/training/trimmed/'
    reject_parth = './sheets/training/rejected/'
    for filename in os.listdir(sheet_path):
        if filename.endswith('.jpg') or filename.endswith('.jepg') or filename.endswith('.png'):
            print(filename)
            segmenter = Segmenter(os.path.join(sheet_path, filename))
            staff_removed = segmenter.remove_staff_lines()
            boxed = segmenter.getSymbols(merge_overlap=True)
            cv2.imwrite(output_path+'boxed{}.png'.format(filename[:-6]), boxed)
            segmenter.saveSymbols(save_origonal=False, format=filename[:-6]+'_note{:03}.png', width=150, path=output_path,reject_path=reject_parth)


if __name__ == '__main__':
    main()