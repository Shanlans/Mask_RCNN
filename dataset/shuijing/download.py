# -*- coding: utf-8 -*-

import os
import shutil
import numpy as np
import hashlib
import uuid
import subprocess
import zipfile
import json
import PIL.Image as Image

import tensorflow as tf

from collections import OrderedDict


def extract_compression_file(fileName, target_folder):
    '''
    Extract zip file and remove system files
    '''

    if not os.path.isdir('temp'):
        os.mkdir('temp')
    if '.zip' == os.path.splitext(fileName)[-1].lower():
        zip_ref = zipfile.ZipFile(fileName, 'r')
        zip_ref.extractall('temp')
        zip_ref.close()
    else:
        raise ValueError('Date compression File should be "Zip" file')

    folders = os.listdir('temp')
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)
    for fold in folders:
        if fold == '__MACOSX' or fold.startswith('.'):
            shutil.rmtree(os.path.join('temp', fold))
        else:
            dirPath = os.path.join(target_folder, fold)
            shutil.move(os.path.join('temp', fold), dirPath)
    shutil.rmtree('temp')


def zipfile_chinese_fix(root_path, dir_name):
    try:
        new_dir_name = dir_name.encode('cp437').decode("gbk")
        os.rename(os.path.join(root_path, dir_name), os.path.join(root_path, new_dir_name))
    except:
        pass


def extract_zipfile(zip_file, target_folder):
    '''
    Extract zip file and remove system files
    '''
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall('temp')
    zip_ref.close()

    folders = os.listdir('temp')
    for fold in folders:
        if fold == '__MACOSX' or fold.startswith('.'):
            shutil.rmtree(os.path.join('temp', fold))
        else:
            shutil.move(os.path.join('temp', fold), target_folder)
    shutil.rmtree('temp')

    for arg, dirnames, names in os.walk(target_folder):
        if '.DS_Store' in names:
            os.remove(os.path.join(arg, '.DS_Store'))

    for dir_name in os.listdir(target_folder):
        zipfile_chinese_fix(target_folder, dir_name)

def is_same_image(image_path1, image_path2):
    hash_md5 = hashlib.md5()
    with open(image_path1, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    md5_1 = hash_md5.hexdigest()

    hash_md5_2 = hashlib.md5()
    with open(image_path2, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5_2.update(chunk)
    md5_2 = hash_md5_2.hexdigest()

    if md5_1 == md5_2:
        return True
    else:
        return False

def merge_metadata(src_metadata_list, dst_metadata_list):
    '''
    src_metadata_list: Source metadata, need to merge
    dst_metadata_list: Destination metadata

    Both src_metadata_list and dst_metadata_list should follow the format such as below:

    [{'ID':1,'Name': [The Class Name]},{'ID':2,'Name': [The Class Name]},{'ID':3,'Name': [The Class Name]}...]

    or

    [{'ID':1,'Name': [The Class Name]},{'ID':3,'Name': [The Class Name]}...]

    '''
    src_dict = {}
    dst_dict = {}

    if src_metadata_list is None:
        # If src dosen't have metadata.json file
        raise ValueError('Please give this dataset "metadata.json" file')
    else:
        for src_metadata in src_metadata_list:
            if src_metadata['ID'] not in src_dict.keys():
                src_dict[src_metadata['ID']] = src_metadata['Name']

    if dst_metadata_list is None:
        tf.logging.info('It is the first time to merge data')
    else:
        for dst_metadata in dst_metadata_list:
            if dst_metadata['ID'] not in dst_dict.keys():
                dst_dict[dst_metadata['ID']] = dst_metadata['Name']

    for src_key, src_value in src_dict.items():
        if src_key in dst_dict.keys():
            if src_value == dst_dict[src_key]:
                # If src key and value are in dst, just merge it directly. E.g. src = { "1" : A, "2":B}, dst = {"1":A, "2":B,...}
                tf.logging.info('Merge Class: ID = [{}] , Name = [{}]'.format(src_key, src_value))
            else:
                # If src key in dst, but src value mismatch dst value which has that key, will raise Error. E.g. src = { "1" : A, "2":C}, dst = {"1":A, "2":B,...}
                raise ValueError(
                    'Src Class ID [{}] and Name [{}] is not match with dst Class ID [{}] and Name [{}]'.format(src_key,
                                                                                                               src_value,
                                                                                                               src_key,
                                                                                                               dst_dict[
                                                                                                                   src_key]))
        elif src_value in dst_dict.values():
            # If src key not in dst, but src value which belongs to the key is also in dst, which means class name is the same, but class number is not
            # E.g. src = { "1" : A, "3":B}, dst = {"1":A, "2":B,...}
            raise ValueError(
                'Src Class Name [{}] is in dst Class which has ID [{}], but not match with dst Class ID'.format(
                    src_value, src_key))
        else:
            # New src key and value
            dst_dict[src_key] = src_value
            tf.logging.info('Update Class: ID = [{}] , Name = [{}]'.format(src_key, src_value))

    dst_dict = OrderedDict(sorted(dst_dict.items(), key=lambda t: t[0], reverse=False))

    merged_dst_list = []

    class_num = len(dst_dict.keys())
    # Create the merged meta_data_list
    for k, v in dst_dict.items():
        merged_dst_list.append({'ID': k, 'Name': v})

    return merged_dst_list, class_num, dst_dict


def get_metadata_info(path):
    with open(path, "r") as data_info:
        metadata_list = json.load(data_info)['items']
    return metadata_list


def write_metadata_info(path, metadata_list):
    update_metadata = {'items': metadata_list}
    with open(path, "w") as data_info:
        json.dump(update_metadata, data_info)



def merger_folder_class(merge_zip_list, target_folder_name,image_size):
    if not os.path.exists(target_folder_name):
        os.makedirs(target_folder_name)

    img_list = []
    label_list = []
    for i, zip_file_path in enumerate(merge_zip_list):
        if zip_file_path.startswith('gs://'):
            local_zip_file_path = 'train_data' + str(i) + '.zip'
            subprocess.check_call([
                'gsutil', '-m', '-q', 'cp', '-r', zip_file_path, local_zip_file_path
            ])
            zip_file_path = local_zip_file_path

        extract_zipfile(zip_file_path, 'tmp_folder' + str(i))
        merged_class_folder_list = os.listdir(target_folder_name)
        class_folder_list = os.listdir('tmp_folder' + str(i))


        for class_folder in class_folder_list:
            if not os.path.isdir(os.path.join(target_folder_name, class_folder)):
                os.mkdir(os.path.join(target_folder_name, class_folder))
            if class_folder not in merged_class_folder_list:  # class folder not exist in merged folder
                for image_name in os.listdir(os.path.join('tmp_folder' + str(i), class_folder)):
                    src_image_path = os.path.join('tmp_folder' + str(i), class_folder, image_name)
                    dst_image_path = os.path.join(target_folder_name, class_folder, os.path.splitext(image_name)[0]+'.png')
                    im = Image.open(src_image_path)
                    im = im.resize(tuple(image_size), Image.NEAREST)
                    if np.array(im).shape[-1] != 3:
                        im = im.convert('RGB')
                    im.save(dst_image_path, "png")
                    img_list.append(dst_image_path)
                    label_list.append(class_folder)
            else:  # class folder is already inside the merged folder, do merge !
                for image_name in os.listdir(os.path.join('tmp_folder' + str(i), class_folder)):
                    src_image_path = os.path.join('tmp_folder' + str(i), class_folder, image_name)
                    dst_image_path = os.path.join(target_folder_name, class_folder, os.path.splitext(image_name)[0]+'.png')
                    if image_name not in os.listdir(os.path.join(target_folder_name, class_folder)):
                        im = Image.open(src_image_path)
                        im = im.resize(tuple(image_size), Image.NEAREST)
                        if np.array(im).shape[-1] != 3:
                            im = im.convert('RGB')
                        im.save(dst_image_path, "png")
                        img_list.append(dst_image_path)
                        label_list.append(class_folder)
                    else:
                        if is_same_image(os.path.join('tmp_folder' + str(i), class_folder, image_name),
                                         os.path.join(target_folder_name, class_folder, image_name)):
                            pass  # do not copy the image
                        else:
                            dst_image_path = os.path.join(target_folder_name, class_folder, str(uuid.uuid4()) + os.path.splitext(image_name)[0]+'.png')
                            im = Image.open(src_image_path)
                            im = im.resize(tuple(image_size), Image.NEAREST)
                            if np.array(im).shape[-1] != 3:
                                im = im.convert('RGB')
                            im.save(dst_image_path, "png")
                            img_list.append(dst_image_path)
                            label_list.append(class_folder)

        shutil.rmtree('tmp_folder' + str(i))
        id_to_className = {}
        class_num = len(os.listdir(target_folder_name))
        for i in range(class_num):
            id_to_className[i]=os.listdir(target_folder_name)[i]
    return (img_list, label_list), class_num,id_to_className


def merger_folder_seg(merge_zip_list, target_folder_name, image_size):
    # Segmentation specified
    target_image_folder = None
    target_mask_folder = None
    if not os.path.isdir(os.path.join(target_folder_name, 'Image')):
        os.makedirs(os.path.join(target_folder_name, 'Image'))
        target_image_folder = os.path.join(target_folder_name, 'Image')
    if not os.path.isdir(os.path.join(target_folder_name, 'Mask')):
        os.makedirs(os.path.join(target_folder_name, 'Mask'))
        target_mask_folder = os.path.join(target_folder_name, 'Mask')

    dst_metadata_list = []
    if 'metadata.json' in os.listdir(target_folder_name):
        dst_metadata_list = get_metadata_info(os.path.join(target_folder_name, 'metadata.json'))

    image_cnt = 0
    image_list = []
    mask_list = []
    for i, zip_file_path in enumerate(merge_zip_list):
        if zip_file_path.startswith('gs://'):
            local_zip_file_path = 'train_data' + str(i) + '.zip'
            print('copy')
            try:
                subprocess.check_call([
                    'gsutil', '-m','-q','cp', '-r', zip_file_path, local_zip_file_path
                ])
            except Exception as e:
                raise e
            print('copy down')
            zip_file_path = local_zip_file_path

        temp_folder_path = 'tmp_folder' + str(i)
        extract_compression_file(zip_file_path, temp_folder_path)

        for root, folder, files in os.walk(temp_folder_path):
            if 'Image' in folder and 'metadata.json' not in os.listdir(root):
                raise ValueError('Please give this [{}] folder "metadata.json" file'.format(root))
            if 'metadata.json' in files:
                src_metadata_list = get_metadata_info(os.path.join(root, 'metadata.json'))

                try:
                    dst_metadata_list, class_num,id_to_className = merge_metadata(src_metadata_list, dst_metadata_list)
                except Exception as e:
                    print(e)

                write_metadata_info(os.path.join(target_folder_name, 'metadata.json'), dst_metadata_list)

            if 'Image' in root:
                for image in files:
                    src_image_path = os.path.join(root, image)
                    src_mask_path = os.path.join(root[:-5], 'Mask', os.path.splitext(image)[0] + '_mask.png')
                    dst_image_path = os.path.join(target_image_folder, 'Image_' + str(image_cnt) + '.jpeg')
                    dst_mask_path = os.path.join(target_mask_folder, 'Image_' + str(image_cnt) + '.png')

                    try:
                        im = Image.open(src_image_path)
                        im = im.resize(tuple(image_size), Image.NEAREST)
                        if np.array(im).shape[-1] != 3:
                            im = im.convert('RGB')
                        im.save(dst_image_path, "jpeg")
                        im.close()
                        if not os.path.isfile(src_mask_path):
                            raise ValueError('Mask file {} not exist'.format(src_mask_path))
                        lbl = Image.open(src_mask_path)
                        lbl = lbl.resize(tuple(image_size), Image.NEAREST)
                        lbl.save(dst_mask_path, "png")
                        lbl.close()
                        image_cnt += 1
                        image_list.append(dst_image_path)
                        mask_list.append(dst_mask_path)
                    except Exception as e:
                        print(e)


        shutil.rmtree('tmp_folder' + str(i))

    tf.logging.info("Final metadata {} ".format(dst_metadata_list))
    return (image_list, mask_list), class_num,id_to_className


def get_data(input_path, output_folder, image_size,job_type):
    # subprocess.check_call(
    #     'apt-get install gcc python-dev python-setuptools; easy_install -U pip ; pip uninstall crcmod ; pip install -U crcmod'
    #     , shell=True)
    if input_path.startswith('Merged;'):
        merge_list = input_path.replace('Merged;', '').split(',')
    elif input_path.startswith('gs://'):
        subprocess.check_call([
                    'gsutil', '-m','-q','cp', '-r', input_path, 'train_data.zip'
                ])
        merge_list = ['train_data.zip']
    else:
        merge_list = input_path.split(',')
    if job_type == 'Classification':
        data_list, class_num,id_to_className = merger_folder_class(merge_list, output_folder, image_size=image_size)
    elif job_type == 'Segmentation':
        data_list, class_num,id_to_className = merger_folder_seg(merge_list, output_folder, image_size=image_size)
    # Just for Mac user ... delete all the .DS_Store files",
    for arg, dirname, names in os.walk(output_folder):
        if '.DS_Store' in names:
            os.remove(os.path.join(arg, '.DS_Store'))
    return data_list, class_num,id_to_className


# Only for Test
if __name__ == "__main__":
    # merge_list = ['aaa.zip', 'bbb.zip']

    li = 'shuijing.zip'

    get_data(li, 'folder_merged',(1024,1024),'Segmentation')
    # merge_list = li.replace('Merged;', '').split(',')

    # merger_folder(merge_list, 'folder_merged')
