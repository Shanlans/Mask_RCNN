import os
import json
import cv2

import numpy as np

from keras.utils import to_categorical
from collections import OrderedDict



def convert_mask_to_annotation(data_folder):
    metadata_file = os.path.join(data_folder,'metadata.json')
    mask_folder = os.path.join(data_folder,'Mask')
    train_folder = os.path.join(data_folder,'train2018')
    val_folder = os.path.join(data_folder,'val2018')
    annotation_folder = os.path.join(data_folder, 'annotations')

    if not os.path.isdir(annotation_folder):
        os.mkdir(annotation_folder)

    with open(metadata_file, "r") as data_info:
        metadata = json.load(data_info)['items']

    class_num = len(metadata)
    class_to_id = OrderedDict()
    for item in metadata:
        key = item['Name']
        val = item['ID']
        class_to_id[val] = key
    print(class_to_id)

    annotation_id = 0
    for i in os.listdir(mask_folder):
        img_path = os.path.join(mask_folder,i)
        dst_path = os.path.join(annotation_folder,i)
        img_name,ext = os.path.splitext(dst_path)
        image = cv2.imread(img_path,0)
        img_to_categorical = to_categorical(image,num_classes=class_num)

        if np.sum(image) == 0 :
            os.remove(os.path.join(train_folder,os.path.splitext(i)[0]+'.jpeg'))
            os.remove(os.path.join(val_folder,os.path.splitext(i)[0]+'.jpeg'))
            os.remove(img_path)
            continue

        if '487' in i:
            print('!!!')

        for cls in range(1,class_num):
            img = img_to_categorical[...,cls]
            if np.sum(img) <10:
                continue
            else:
                dst_name = img_name+'_'+class_to_id[cls]+'_'+str(annotation_id)+ext
                cv2.imwrite(dst_name,img)
                # cv2.imshow(class_to_id[cls], img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                annotation_id+=1

    print('Total {} annotations generated'.format(annotation_id))










if __name__ == '__main__':
    convert_mask_to_annotation('folder_merged')







