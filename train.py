import tensorflow as tf
import os
import time
import numpy as np

from SRCNN.util import Time
from SRCNN.util import ImageBatch
from SRCNN.util import psnr

def training(SRCNN, config):
    training_ratio = 1
    main_data_path = '.'
    '''
    DATA SET'S PATH
    '''
    train_inputs_path = '{}/data/train_91_input/*.npy'.format(main_data_path)
    train_labels_path = '{}/data/train_91_label/*.npy'.format(main_data_path)
    test_labels_path = '{}/data/Set5/y_ch/*.npy'.format(main_data_path)
    test_inputs_path = '{}/data/Set5/y_ch_2x/*.npy'.format(main_data_path)
    '''
    TRAIN SET(91) and shuffle
    '''
    train_inputs_batch = ImageBatch(train_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    train_labels_batch = ImageBatch(train_labels_path, training_ratio=training_ratio, on_sort=True, ext='npy')

    shuffle_indicese = list(range(train_inputs_batch.N_TRAIN_DATA))
    np.random.shuffle(shuffle_indicese)
    train_inputs_batch.train_shuffle(shuffle_indicese)
    train_labels_batch.train_shuffle(shuffle_indicese)
    '''
    TEST SET(SET 9) & preprocessing
    '''
    test_labels_batch = ImageBatch(test_labels_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_inputs_batch = ImageBatch(test_inputs_path, training_ratio=training_ratio, on_sort=True, ext='npy')
    test_labels = test_labels_batch.next_batch(batch_size=5)
    test_inputs = test_inputs_batch.next_batch(batch_size=5)

    avg_bicubic_psnr_y_ch = 0
    for i in range(len(test_labels)):
        _psnr = psnr(test_labels[i], test_inputs[i], peak=1)
        avg_bicubic_psnr_y_ch += _psnr / 5

    training_epoch = config.training_epoch
    batch_size = config.batch_size
    n_data = train_inputs_batch.N_TRAIN_DATA
    total_batch = n_data // batch_size if n_data % batch_size == 0 else (n_data // batch_size) + 1

    SRCNN.neuralnet()
    SRCNN.optimize(config)
    SRCNN.summary()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=tf_config)
    writer = tf.summary.FileWriter('./model/srcnn_result', sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    print("Total the number of Data : " + str(train_inputs_batch.N_TRAIN_DATA))
    print("Total Step per 1 Epoch: {}".format(total_batch))
    print("The number of Iteration: {}".format(training_epoch * total_batch))

    step = 0
    for epoch in range(training_epoch):
        avg_cost = 0
        avg_vdsr_psnr_y_ch = 0
        for i in range(total_batch):
            start = time.time()
            batch_x = train_inputs_batch.next_batch(batch_size, num_thread=8)
            batch_x = np.expand_dims(batch_x, axis=-1)
            batch_y = train_labels_batch.next_batch(batch_size, num_thread=8)
            batch_y = np.expand_dims(batch_y, axis=-1)

            summaries, _cost, _ = sess.run([SRCNN.summaries, SRCNN.cost, SRCNN.optimizer],
                                           feed_dict={SRCNN.X: batch_x, SRCNN.Y: batch_y})
            writer.add_summary(summaries, step)

            avg_cost += _cost / total_batch
            end = time.time()
            step += 1

            if epoch % 10 ==0 and i == 20:
                Time.require_time(start, end, count=training_epoch * total_batch - step)

        if epoch % 10 == 0:
            '''
           Evaluate VDSR performance
           '''
            for index in range(5):
                '''
               Y ch test average PSNR
               '''
                label_y = (test_labels[index].copy()*255).astype(np.uint8)
                input_y = test_inputs[index].copy()

                shape = input_y.shape
                result_input_y = input_y.reshape((1, *shape, 1))
                result_input_y = sess.run(SRCNN.output, feed_dict={SRCNN.X: result_input_y})
                result_input_y = result_input_y.reshape(shape).astype(np.float32)
                result_input_y *= 255
                result_input_y = result_input_y.astype(np.uint8)

                _psnr = psnr(label_y, result_input_y, peak=255)
                print(_psnr)
                avg_vdsr_psnr_y_ch += _psnr / 5

            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost),
                  '\nY_Ch AVG PSNR:: Bicubic: {:.9f} || VDSR: {:.9f}'.format(avg_bicubic_psnr_y_ch, avg_vdsr_psnr_y_ch))

        np.random.shuffle(shuffle_indicese)
        train_inputs_batch.train_shuffle(shuffle_indicese)
        train_labels_batch.train_shuffle(shuffle_indicese)

    print("학습 완료!")
    save_path = '{}/model/SRCNN.ckpt'.format(os.path.abspath('.'))
    saver.save(sess, save_path)
    print("세이브 완료")
