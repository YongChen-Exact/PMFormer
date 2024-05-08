import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary


def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3

    return rv, myo, lv


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def inference(fold):
    label_list = sorted(glob.glob(
        os.path.join(r"G:\CY\CodesOfCy\new\PMFormer-main\DATASET\PMFormer_raw\PMFormer_raw_data\Task01_ACDC\labelsTs",
                     '*nii.gz')))
    dest_file = r"G:\CY\results\Third\ACDC\fold_4_UNETR_PP_multiresblock_cbam_mixffn_mulinput_ds\model_best"

    infer_list = sorted(glob.glob(
        (os.path.join(dest_file, '*nii.gz'))))

    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_rv = []
    Dice_myo = []
    Dice_lv = []
    Dice_avg = []

    hd_rv = []
    hd_myo = []
    hd_lv = []
    hd_avg = []

    file = dest_file
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + '/dice_pre.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path)
        print(infer_path)
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, spacing = read_nii(label_path)
        infer, spacing = read_nii(infer_path)
        label_rv, label_myo, label_lv = process_label(label)
        infer_rv, infer_myo, infer_lv = process_label(infer)

        Dice_tmp = []
        Dice_rv.append(dice(infer_rv, label_rv))
        Dice_myo.append(dice(infer_myo, label_myo))
        Dice_lv.append(dice(infer_lv, label_lv))

        Dice_tmp.append(dice(infer_rv, label_rv))
        Dice_tmp.append(dice(infer_myo, label_myo))
        Dice_tmp.append(dice(infer_lv, label_lv))

        Dice_avg.append(np.mean(Dice_tmp))

        hd_tmp = []

        hd_rv.append(hd(infer_rv, label_rv))
        hd_myo.append(hd(infer_myo, label_myo))
        hd_lv.append(hd(infer_lv, label_lv))

        hd_tmp.append(hd(infer_rv, label_rv))
        hd_tmp.append(hd(infer_myo, label_myo))
        hd_tmp.append(hd(infer_lv, label_lv))

        hd_avg.append(np.mean(hd_tmp))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        # fw.write('*'*20+'\n')
        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*' * 20 + '\n')

    # fw.write('*'*20+'\n')
    # fw.write('Mean_hd\n')
    # fw.write('hd_rv'+str(np.mean(hd_rv))+'\n')
    # fw.write('hd_myo'+str(np.mean(hd_myo))+'\n')
    # fw.write('hd_lv'+str(np.mean(hd_lv))+'\n')
    # fw.write('*'*20+'\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_rv)) + '\n')
    fw.write('Dice_myo' + str(np.mean(Dice_myo)) + '\n')
    fw.write('Dice_lv' + str(np.mean(Dice_lv)) + '\n')
    fw.write('Mean_HD\n')
    fw.write('HD_rv' + str(np.mean(hd_rv)) + '\n')
    fw.write('HD_myo' + str(np.mean(hd_myo)) + '\n')
    fw.write('HD_lv' + str(np.mean(hd_lv)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    avg_hd = []
    avg_hd.append(np.mean(hd_rv))
    avg_hd.append(np.mean(hd_myo))
    avg_hd.append(np.mean(hd_lv))
    fw.write('avg_hd:' + str(np.mean(avg_hd)) + '\n')

    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    fw.write('HD:' + str(np.mean(avg_hd)) + '\n')

    fw.write('sample_dsc:' + str(Dice_avg) + '\n')
    fw.write('sample_hd:' + str(hd_avg) + '\n')

    print('done')


if __name__ == '__main__':
    inference(1)
