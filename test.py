from psnr import PSNR_TEST, scale_psnr
import psnr
import Model
import os
import DataLoader
import numpy as np
import tensorflow as tf

test_folder_list='D:/SRCNN/SRCNN_KERAS/test_file/set14_x2'
check_list='./checkpoints'
max_psnr=0
epoch_list=0

for root,dirs, files in os.walk(check_list):
    for fn in files:
        full_filename=os.path.join(root, fn)
        full_filename=os.path.basename(full_filename)

        epoch = (full_filename.split('-')[2])

        full_filename = os.path.join(check_list,full_filename)
        srcnn_model = Model.test_model()
        srcnn_model.load_weights(full_filename)

        avg_psnr=0
        avg_bicubic=0
        input_cnt=0
        test_cnt=0
        gt_cnt=0

        input_y=[]
        gt_y=[]

        img_list=DataLoader.get_img_list(test_folder_list)
        print(len(img_list))
        length_file=len(img_list)

        for i in range(len(img_list)):
            input_list, gt_list, scale_list=DataLoader.get_test_image(img_list, i, 1)
            input_y=np.array(input_list[0])
            gt_y=np.array(gt_list[0])

            input_y=np.resize(input_y,(1, input_y.shape[0], input_y.shape[1],1))
            gt_y=np.resize(gt_y,(1,gt_y.shape[0], gt_y.shape[1], 1))

            input_y=input_y.astype(float) / 255.
            predicted=srcnn_model.predict(input_y) * 255.

            input_y=input_y.astype(float) * 255.

            gt_y=np.clip(gt_y, 0, 255)
            predicted=np.clip(predicted, 0, 255)
            input_y=np.clip(input_y, 0, 255)

            scale = 2

            gt_y_psnr , predicted_psnr, input_y_psnr = scale_psnr(gt_y, predicted, input_y, scale)

            psnr=PSNR_TEST(gt_y_psnr, predicted_psnr)
            bicubic_psnr = PSNR_TEST(gt_y_psnr, input_y_psnr)

            sess=tf.Session()
            result=sess.run(psnr)
            bicubic_result=sess.run(bicubic_psnr)

            avg_psnr += result
            avg_bicubic += bicubic_result

            predicted = np.reshape(predicted, (predicted.shape[1], predicted.shape[2]))
            input_y = np.reshape(input_y, (input_y.shape[1], input_y.shape[2]))
            gt_y = np.reshape(gt_y, (gt_y.shape[1], gt_y.shape[2]))

            input_count = str(input_cnt)
            test_count = str(test_cnt)
            gt_count = str(gt_cnt)

            gt_y = gt_y.astype('uint8')
            input_y = input_y.astype('uint8')
            predicted = predicted.astype('uint8')

            # misc.imsave(os.path.join('./Result/gt',epoch+'_'+'gt'+gt_count+'.png'),gt_y)
            gt_cnt += 1
            # misc.imsave(os.path.join('./Result/bicubic',epoch+'_'+'bicubic'+input_count+'.png'),input_y)
            input_cnt += 1
            # misc.imsave(os.path.join('./Result/test',epoch+'_'+'test'+test_count+'.png'),predicted)
            test_cnt += 1

        avg_psnr=avg_psnr / length_file
        avg_bicubic=avg_bicubic / length_file
        print('{0}_epochs bicubic psnr for proposed method is {1:.3f}dB'.format(epoch, avg_bicubic))
        print('{0}_epochs average psnr for proposed method is {1:.3f}dB'.format(epoch, avg_psnr))

        if max_psnr < avg_psnr:
            max_psnr = avg_psnr
            epoch_list = epoch
            print("max_psnr:{0:.3f}, avg_psnr::{1:.3f}".format(max_psnr, avg_psnr))

print('max_level:{0}epoch, {1:.3f}dB'.format(epoch_list,max_psnr))
