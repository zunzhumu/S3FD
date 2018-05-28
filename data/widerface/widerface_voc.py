# -*- coding: utf-8 -*-
"""
Created on 17-5-27

@author: zly 
"""
from skimage import io
import shutil
import random
import os
import string

headstr = """\
<annotation>
    <folder>VOC2007</folder>
    <filename>%06d.jpg</filename>
    <source>
        <database>My Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <image>flickr</image>
        <flickrid>NULL</flickrid>
    </source>
    <owner>
        <flickrid>NULL</flickrid>
        <name>company</name>
    </owner>
    <size>
        <width>%d</width>
        <height>%d</height>
        <depth>%d</depth>
    </size>
    <segmented>0</segmented>
"""
objstr = """\
    <object>
        <name>%s</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>%d</xmin>
            <ymin>%d</ymin>
            <xmax>%d</xmax>
            <ymax>%d</ymax>
        </bndbox>
    </object>
"""

tailstr = '''\
</annotation>
'''

def all_path(filename):
    return os.path.join('widerface', filename)

def writexml(idx, head, bbxes, tail):
    filename = all_path("Annotations/%06d.xml" % (idx))
    f = open(filename, "w")
    f.write(head)
    for bbx in bbxes:
        f.write(objstr % ('face', bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]))
    f.write(tail)
    f.close()


def clear_dir():
    if shutil.os.path.exists(all_path('Annotations')):
        shutil.rmtree(all_path('Annotations'))
    if shutil.os.path.exists(all_path('ImageSets')):
        shutil.rmtree(all_path('ImageSets'))
    if shutil.os.path.exists(all_path('JPEGImages')):
        shutil.rmtree(all_path('JPEGImages'))

    shutil.os.mkdir(all_path('Annotations'))
    shutil.os.makedirs(all_path('ImageSets/Main'))
    shutil.os.mkdir(all_path('JPEGImages'))


def excute_datasets(idx, datatype):
    f = open(all_path('ImageSets/Main/' + datatype + '.txt'), 'a')
    f_bbx = open(all_path('wider_face_split/wider_face_' + datatype + '_bbx_gt.txt'), 'r')

    while True:
        filename = string.strip(f_bbx.readline(), '\n')
        if not filename:
            break
        im = io.imread(all_path('WIDER_' + datatype + '/images/'+filename))
        head = headstr % (idx, im.shape[1], im.shape[0], im.shape[2])
        nums = string.strip(f_bbx.readline(), '\n')
        bbxes = []
        for ind in xrange(string.atoi(nums)):
            bbx_info = string.split(string.strip(f_bbx.readline(), ' \n'), ' ')
            bbx = [string.atoi(bbx_info[i]) for i in range(len(bbx_info))]
            #x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose
            if bbx[7]==0:
                bbxes.append(bbx)
        writexml(idx, head, bbxes, tailstr)
        shutil.copyfile(all_path('WIDER_' + datatype + '/images/'+filename), all_path('JPEGImages/%06d.jpg' % (idx)))
        f.write('%06d\n' % (idx))
        idx +=1
    f.close()
    f_bbx.close()
    return idx


# 打乱样本
def shuffle_file(filename):
    f = open(filename, 'r+')
    lines = f.readlines()
    random.shuffle(lines)
    f.seek(0)
    f.truncate()
    f.writelines(lines)
    f.close()


if __name__ == '__main__':
    clear_dir()
    idx = 1
    idx = excute_datasets(idx, 'train')
    idx = excute_datasets(idx, 'val')
