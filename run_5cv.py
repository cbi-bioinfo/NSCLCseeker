from sklearn.model_selection import train_test_split
import tensorflow  as tf
import numpy as np
import pandas as pd
import os
#import cv2
from sklearn import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
import random
import sys

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

ver = sys.argv[1]
filename = 'adc_scc_lcc_ft_no_fix_m50_' + ver
resDirName = 'results/'

createFolder('./models/' + filename)
createFolder('./' + resDirName +filename)
createFolder('./' + resDirName + filename+'/pred_png')
createFolder('./' + resDirName + filename+'/test_x_png')
createFolder('./' + resDirName + filename+'/test_y_png')

os.environ["CUDA_VISIBLE_DEVICES"]="0"
config=tf.ConfigProto()
config.intra_op_parallelism_threads=44
config.inter_op_parallelism_threads=44
config.gpu_options.allow_growth=True

TEST_SIZE = 0.2
learning_rate = 1e-5
seg_training_epochs = 2000
class_training_epochs = 2000
finetuning_epochs = 1000
batch_size = 500

X = tf.placeholder(tf.float32, [None, 128, 128, 1], name='X')
Y = tf.placeholder(tf.float32, [None, 128, 128, 1], name='Y')
Z = tf.placeholder(tf.float32, [None, 3], name='Z')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
keep_prob_class = tf.placeholder(tf.float32, name='keep_prob_class')

x_train = []
x_test = []
y_train = []
y_test = []
z_train = []
z_test = []

total_x_train = []
total_z_train = []

df = pd.read_csv('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/train_num_subtype_' + ver + '.csv')
df_num = df['patient_num']
z_train_patient_label = df['patient_label']
encoding = pd.get_dummies(z_train_patient_label)
z_train = encoding.to_numpy()

df_test = pd.read_csv('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/test_num_subtype_' + ver + '.csv')
df_num_test = df_test['patient_num']
z_test_patient_id = df_num_test
z_test_patient_label = df_test['patient_label']
encoding_test = pd.get_dummies(z_test_patient_label)
z_test = encoding_test.to_numpy()
z_test_patient_id = z_test_patient_id.astype(int)

x_train=np.load('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/x_train_' + ver + '.npy')
y_train=np.load('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/y_train_' + ver + '.npy')
for a in range(0,y_train.shape[0]) :
        total_x_train.append(x_train[a,:])
        total_z_train.append(z_train[a,:])
print('train set ends')

x_test=np.load('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/x_test_' + ver + '.npy')
y_test=np.load('/data/projects/lung_ct/kong/data_v2/LUNG1_5cv_128_subtype3_m50/y_test_' + ver +  '.npy')
print('test set ends')

### Add LUNG3 Dataset ###
df_2 = pd.read_csv('/data/projects/lung_ct/kong/csv_file/patient_subtype_adc_preprocess_LUNG3.csv',dtype={'num':float})
df_2_num = df_2['patient_num']
add_train_num = df_2_num.astype(int)
train_count = -1
for num in add_train_num:
	train_count+=1
	x_data = np.load('/data/projects/lung_ct/kong/data_v2/LUNG3_data_resize128_x/LUNG3_resize128_x_'+str(num)+'.npy')
	for a in range(0, x_data.shape[0]):
		total_x_train.append(x_data[a,:])
		total_z_train.append([1,0,0])
######

x_test = np.array(x_test)
y_test = np.array(y_test)
z_train = np.array(z_train)
z_test = np.array(z_test)
total_x_train = np.array(total_x_train)
total_z_train = np.array(total_z_train)

print('x_test.shape : ' + str(x_test.shape))
print('y_test.shape : ' + str(y_test.shape))


smooth = 1.
threshold=0.5
is_pretrain = 1

def dice_coef_func(y_true, y_pred):
		ground_truth_area = tf.reduce_sum(y_true, axis=[1,2,3])
		prediction_area = tf.reduce_sum(y_pred, axis=[1,2,3])
		intersection_area = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
		combined_area = ground_truth_area + prediction_area
		dice = tf.reduce_mean((2*intersection_area + smooth)/(combined_area + smooth))
		ga =tf.cast(ground_truth_area, tf.float32)
		pa = tf.cast(prediction_area, tf.float32)
		ia = tf.cast(intersection_area, tf.float32)
		ca = tf.cast(combined_area, tf.float32)
		return dice

def dice_loss(y_true, y_pred):
	dice = dice_coef_func(y_true, y_pred)
	return 1-dice

def iou_func(y_true, y_pred):
		ground_truth_area = tf.reduce_sum(y_true, axis=[1,2,3])
		prediction_area = tf.reduce_sum(y_pred, axis=[1,2,3])
		intersection_area = tf.reduce_sum(y_true*y_pred, axis=[1,2,3])
		union_area = (ground_truth_area	+ prediction_area	- intersection_area)
		iou = tf.reduce_mean((intersection_area + smooth)/(union_area + smooth))
		return iou

def cce_loss(z_true, z_pred):
	cce = -tf.reduce_sum(z_true*tf.log(z_pred + 1e-10), axis = 1)
	cce_loss = tf.reduce_mean(cce)
	return cce_loss

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

### model ###

with tf.name_scope("u-net") :
        vx = tf.Variable(10.0, name="vx")
        conv1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', name = 'u-net1')(X)
        conv1 = tf.keras.layers.Conv2D(32, (3,3), padding='same', name = 'u-net2')(conv1)
        drop1 = tf.nn.dropout(conv1, keep_prob)
        bn1 = tf.layers.batch_normalization(drop1)
        bn1 = tf.nn.relu(bn1)

        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name = 'u-net3')(bn1)
        conv2 = tf.keras.layers.Conv2D(64, (3,3), padding='same',name = 'u-net4')(pool1)
        conv2 = tf.keras.layers.Conv2D(64, (3,3), padding='same',name = 'u-net5')(conv2)
        drop2 = tf.nn.dropout(conv2, keep_prob)
        bn2 = tf.layers.batch_normalization(drop2)
        bn2 = tf.nn.relu(bn2)

        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name = 'u-net6')(bn2)
        conv3 = tf.keras.layers.Conv2D(128, (3,3), padding='same',name = 'u-net7')(pool2)
        conv3 = tf.keras.layers.Conv2D(128, (3,3), padding='same',name = 'u-net8')(conv3)
        drop3 = tf.nn.dropout(conv3, keep_prob)
        bn3 = tf.layers.batch_normalization(drop3)
        bn3 = tf.nn.relu(bn3)

        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name = 'u-net9')(bn3)
        conv4 = tf.keras.layers.Conv2D(256, (3,3), padding='same',name = 'u-net10')(pool3)
        conv4 = tf.keras.layers.Conv2D(256, (3,3), padding='same',name = 'u-net11')(conv4)
        drop4 = tf.nn.dropout(conv4, keep_prob)
        bn4 = tf.layers.batch_normalization(drop4)
        bn4 = tf.nn.relu(bn4)

        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2,2),name = 'u-net12')(bn4)
        conv5 = tf.keras.layers.Conv2D(512, (3,3), padding='same',name = 'u-net13')(pool4)
        conv5 = tf.keras.layers.Conv2D(512, (3,3), padding='same',name = 'u-net14')(conv5)
        drop5 = tf.nn.dropout(conv5, keep_prob)
        bn5 = tf.layers.batch_normalization(drop5)
        bn5 = tf.nn.relu(bn5, name='bn5')

        up6 = tf.keras.layers.concatenate([tf.layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(bn5), bn4], axis=3)
        conv6 = tf.keras.layers.Conv2D(256, (3,3), padding='same')(up6)
        conv6 = tf.keras.layers.Conv2D(256, (3,3), padding='same')(conv6)
        drop6 = tf.nn.dropout(conv6, keep_prob)
        bn6 = tf.layers.batch_normalization(drop6)
        bn6 = tf.nn.relu(bn6)

        up7 = tf.keras.layers.concatenate([tf.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(bn6), bn3], axis=3)
        conv7 = tf.keras.layers.Conv2D(128, (3,3), padding='same')(up7)
        conv7 = tf.keras.layers.Conv2D(128, (3,3), padding='same')(conv7)
        drop7 = tf.nn.dropout(conv7, keep_prob)
        bn7 = tf.layers.batch_normalization(drop7)
        bn7 = tf.nn.relu(bn7)

        up8 = tf.keras.layers.concatenate([tf.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(bn7), bn2], axis=3)
        conv8 = tf.keras.layers.Conv2D(64, (3,3), padding='same')(up8)
        conv8 = tf.keras.layers.Conv2D(64, (3,3), padding='same')(conv8)
        drop8 = tf.nn.dropout(conv8, keep_prob)
        bn8 = tf.layers.batch_normalization(drop8)
        bn8 = tf.nn.relu(bn8)

        up9 = tf.keras.layers.concatenate([tf.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(bn8), bn1], axis=3)
        conv9 = tf.keras.layers.Conv2D(32, (3,3), padding='same')(up9)
        conv9 = tf.keras.layers.Conv2D(32, (3,3), padding='same')(conv9)
        drop9 = tf.nn.dropout(conv9, keep_prob)
        bn9 = tf.layers.batch_normalization(drop9)
        bn9 = tf.nn.relu(bn9)

        conv10_seg = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid', name='segmentation')(bn9)


seg_loss = tf.reduce_mean(dice_loss(Y, conv10_seg))
#ae_loss = tf.reduce_mean(tf.squared_difference(conv10_ae, X))
#ae_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(ae_loss)
seg_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(seg_loss)

iou = iou_func(Y, tf.round(tf.cast(conv10_seg, tf.float32)))
dice  = dice_coef_func(Y, tf.round(tf.cast(conv10_seg, tf.float32)))

subtype_inputs = tf.layers.Flatten(name = "subtype_inputs")(bn5)

dense1 = tf.layers.Dense(64, activation='relu', name="class_layer1")(subtype_inputs)
dropfc1 = tf.nn.dropout(dense1, keep_prob_class, name='class_layer2')
dense2 = tf.layers.Dense(32, activation='relu', name = 'class_layer3')(dropfc1)
dropfc2 = tf.nn.dropout(dense2, keep_prob_class, name='class_layer4')
subtype_outputs = tf.layers.Dense(3, activation='softmax', name = 'subtype_outputs')(dropfc2)

class_loss = tf.reduce_mean(cce_loss(Z, subtype_outputs), name="loss")

correct_prediction = tf.equal(tf.argmax(subtype_outputs, 1), tf.argmax(Z, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
class_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(class_loss)

total_loss = seg_loss + class_loss
total_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

epoch_ae_score = pd.DataFrame(columns=['epoch', 'ae_loss',  'test_ae_loss'])
epoch_seg_score = pd.DataFrame(columns=['epoch', 'seg_loss', 'seg_dice', 'seg_iou',  'test_seg_loss', 'test_seg_dice', 'test_seg_iou'])
epoch_class_score = pd.DataFrame(columns=['epoch', 'class_loss', 'acc', 'test_class_loss', 'test_total_acc','test_acc_adc', 'test_acc_lcc', 'test_acc_scc','test_auc' ,  'test_f1_score'])

max_dice = 0
max_total_acc = 0
max_auc = 0
max_ae_loss = 10000

saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
	for epoch in range(class_training_epochs): #subtype classification part
		avg_class_loss = 0
		avg_acc = 0
		total_batch = np.ceil(total_x_train.shape[0]/batch_size)
		#print(total_batch)
		for current_batch_index in range(0, total_x_train.shape[0], batch_size):
			current_class_score = []
			current_X = total_x_train[current_batch_index:current_batch_index+batch_size, :]
			current_Z = total_z_train[current_batch_index:current_batch_index+batch_size, :]
			feed_dict = {X : current_X, Z : current_Z, keep_prob:1.0, keep_prob_class: 0.5}

			class_c, _, ba = sess.run([class_loss, class_optimizer, accuracy], feed_dict=feed_dict)
			avg_class_loss += class_c/total_batch
			avg_acc += ba/total_batch
			
		if epoch % 10 == 0 :
			print(' ----- Epoch : '+str(epoch+1))
			test_class_loss = 0
			test_acc_list = []
			test_auc_list = []
			z_prediction = np.empty((0, 3), float)
			z_pred_sm = np.empty((0, 3), float)
			total_test_batch = np.ceil(x_test.shape[0]/batch_size)
			#print(total_test_batch)
			for test_batch_index in range(0, x_test.shape[0], batch_size):
				test_current_X = x_test[test_batch_index : test_batch_index+batch_size, :]
				test_current_Z = z_test[test_batch_index :  test_batch_index+batch_size, :]
				feed_dict = {X : test_current_X, Z : test_current_Z, keep_prob:1.0, keep_prob_class: 1.0}
				class_c = sess.run([class_loss], feed_dict=feed_dict)
				test_class_loss += class_c/total_test_batch
				current_z_pred = sess.run(subtype_outputs, feed_dict={X : test_current_X, Z : test_current_Z, keep_prob:1.0, keep_prob_class:1.0})
				z_prediction = np.append(z_prediction, np.round(current_z_pred), axis = 0)
				z_pred_sm = np.append(z_pred_sm, current_z_pred, axis = 0)
			
			total_z_prediction = tf.equal(tf.argmax(z_prediction, 1), tf.argmax(z_test, 1))
			acc = metrics.accuracy_score(np.argmax(z_test, 1), np.argmax(z_prediction, 1))
			cm = metrics.confusion_matrix(np.argmax(z_test, 1), np.argmax(z_prediction, 1))
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			acc_for_each = cm.diagonal()
			auc = metrics.roc_auc_score(z_test, z_prediction, average = 'weighted', multi_class='ovr')
			f1_score = metrics.f1_score(np.argmax(z_test, 1), np.argmax(z_prediction, 1), average = 'weighted')

			print('[ Train Class Loss = {:.4f} Train Acc = {:.4f} ]'.format(avg_class_loss, avg_acc))
			print('[ Test Class Loss = {:.4f}  Test Total Acc : {:.4f}  Test Acc_adc : {:.4f}  Test Acc_lcc : {:.4f}  Test Acc_scc : {:.4f}  Test Auc : {:.4f} Test F1 Score : {:.4f}]\n'.format(float(test_class_loss), float(acc), float(acc_for_each[0]), float(acc_for_each[1]), float(acc_for_each[2]), float(auc), float(f1_score)))

			if acc >= max_total_acc :
				max_epoch = epoch+1
				max_class_loss = test_class_loss
				max_total_acc = acc
				max_acc_adc = acc_for_each[0]
				max_acc_lcc = acc_for_each[1]
				max_acc_scc = acc_for_each[2]
				##
				max_auc = auc
				##
				max_f1_score = f1_score
				max_z_prediction = z_prediction
				max_z_pred_sm = z_pred_sm

			current_class_score.extend([epoch+1, avg_class_loss, avg_acc, test_class_loss, acc,acc_for_each[0], acc_for_each[1], acc_for_each[2], auc, f1_score])
			epoch_class_score.loc[len(epoch_class_score)] = current_class_score
			epoch_class_score.to_csv("./"+ resDirName + filename+"/"+filename+"_epoch_class_score.csv", index=False)
		if epoch == class_training_epochs -1:
			print('Classification pretraining Ends...')

	for epoch in range(seg_training_epochs): #segmentation part
		avg_seg_loss = 0
		avg_dice = 0
		avg_iou = 0
		total_batch = np.ceil(x_train.shape[0]/batch_size)
		for current_batch_index in range(0, x_train.shape[0], batch_size):
			current_seg_score = []
			current_X = x_train[current_batch_index:current_batch_index+batch_size, :]
			current_Y = y_train[current_batch_index:current_batch_index+batch_size, :]
			feed_dict = {X : current_X, Y : current_Y, keep_prob:0.5}

			seg_c,  _, bd, bi = sess.run([seg_loss, seg_optimizer, dice, iou], feed_dict=feed_dict)
	
			avg_seg_loss += seg_c/total_batch
			avg_dice += bd/total_batch
			avg_iou += bi/total_batch
			
		if epoch % 20 == 0 :
			print(' ----- Epoch : '+str(epoch+1))
			test_seg_loss = 0
			test_dice = 0
			test_iou = 0
			test_dice_list = []
			y_prediction = np.empty((0, 128, 128, 1), float)
			total_test_batch = np.ceil(x_test.shape[0]/batch_size)
			# print(total_test_batch)
			for test_batch_index in range(0, x_test.shape[0], batch_size):
				test_current_X = x_test[test_batch_index : test_batch_index+batch_size, :]
				test_current_Y = y_test[test_batch_index :  test_batch_index+batch_size, :]
				feed_dict = {X : test_current_X, Y : test_current_Y,  keep_prob:1}
				seg_c, bd, bi= sess.run([seg_loss,dice, iou], feed_dict=feed_dict)
				test_seg_loss += seg_c/total_test_batch
				test_dice += bd/total_test_batch
				test_iou += bi/total_test_batch
				test_dice_list.append(bd)

				current_y_pred = sess.run(conv10_seg, feed_dict={X : test_current_X, Y : test_current_Y, keep_prob: 1.0})
				y_prediction = np.append(y_prediction, current_y_pred, axis = 0)

			print('[ Train Seg Loss = {:.4f} Train Dice : {:.4f} Train Iou = {:.4f} ]'.format(avg_seg_loss,  avg_dice, avg_iou))
			print('[ Test Seg Loss = {:.4f} Test Dice : {:.4f} Test Iou : {:.4f} ]\n'.format(test_seg_loss, test_dice, test_iou))

			if test_dice >= max_dice :
				max_epoch = epoch+1
				max_dice = test_dice
				max_iou  = test_iou
				max_seg_loss = test_seg_loss
				max_y_prediction = y_prediction
				max_dice_list = test_dice_list

			current_seg_score.extend([epoch+1, avg_seg_loss, avg_dice, avg_iou, test_seg_loss, test_dice, test_iou])
			#epoch_seg_score.loc[len(epoch_seg_score)] = current_seg_score
			#epoch_seg_score.to_csv("./"+ resDirName+filename+"/"+filename+"_epoch_seg_score.csv", index=False)
		if epoch == seg_training_epochs -1:
			print('Segmentation Training Ends...')


	for epoch in range(finetuning_epochs): #fine tuning part
		avg_class_loss = 0
		avg_acc = 0	
		avg_seg_loss = 0
		avg_dice = 0
		avg_iou = 0
		avg_total_loss = 0
		total_batch = np.ceil(x_train.shape[0]/batch_size)
		#print(total_batch)
		for current_batch_index in range(0, x_train.shape[0], batch_size):
			current_class_score = []
			current_X = x_train[current_batch_index:current_batch_index+batch_size, :]
			current_Y = y_train[current_batch_index:current_batch_index+batch_size, :]
			current_Z = z_train[current_batch_index:current_batch_index+batch_size, :]
			feed_dict = {X : current_X, Y : current_Y, Z : current_Z, keep_prob:0.5, keep_prob_class: 0.5}

			total_c, class_c, _, ba, seg_c, bd, bi = sess.run([total_loss, class_loss, total_optimizer, accuracy, seg_loss, dice, iou], feed_dict=feed_dict)
			avg_class_loss += class_c/total_batch
			avg_acc += ba/total_batch
			avg_seg_loss += seg_c/total_batch
			avg_dice += bd/total_batch
			avg_iou += bi/total_batch
			avg_total_loss += total_c/total_batch
	
		if epoch % 10 == 0 :
			print(' ----- Epoch : '+str(epoch+1))
			test_class_loss = 0
			test_acc_list = []
			test_auc_list = []
			z_prediction = np.empty((0, 3), float)
			z_pred_sm = np.empty((0, 3), float)
			test_seg_loss = 0
			test_dice = 0
			test_iou = 0
			test_dice_list = []
			y_prediction = np.empty((0, 128, 128, 1), float)
			total_test_batch = np.ceil(x_test.shape[0]/batch_size)

			for test_batch_index in range(0, x_test.shape[0], batch_size):
				test_current_X = x_test[test_batch_index : test_batch_index+batch_size, :]
				test_current_Y = y_test[test_batch_index :  test_batch_index+batch_size, :]
				test_current_Z = z_test[test_batch_index :  test_batch_index+batch_size, :]
				feed_dict = {X : test_current_X, Z : test_current_Z, Y : test_current_Y, keep_prob:1.0, keep_prob_class: 1.0}
				class_c,seg_c, bd, bi, current_z_pred, current_y_pred = sess.run([class_loss, seg_loss,dice, iou,subtype_outputs, conv10_seg], feed_dict=feed_dict)
				test_class_loss += class_c/total_test_batch
				z_prediction = np.append(z_prediction, np.round(current_z_pred), axis = 0)
				z_pred_sm = np.append(z_pred_sm, current_z_pred, axis = 0)


				test_seg_loss += seg_c/total_test_batch
				test_dice += bd/total_test_batch
				test_iou += bi/total_test_batch
				test_dice_list.append(bd)

				y_prediction = np.append(y_prediction, current_y_pred, axis = 0)

			print('[ Train Seg Loss = {:.4f} Train Dice : {:.4f} Train Iou = {:.4f} ]'.format(avg_seg_loss,  avg_dice, avg_iou))
			print('[ Test Seg Loss = {:.4f} Test Dice : {:.4f} Test Iou : {:.4f} ]\n'.format(test_seg_loss, test_dice, test_iou))
			
			total_z_prediction = tf.equal(tf.argmax(z_prediction, 1), tf.argmax(z_test, 1))
			acc = metrics.accuracy_score(np.argmax(z_test, 1), np.argmax(z_prediction, 1))
			cm = metrics.confusion_matrix(np.argmax(z_test, 1), np.argmax(z_prediction, 1))
			cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			acc_for_each = cm.diagonal()
			auc = metrics.roc_auc_score(z_test, z_prediction, average = 'weighted', multi_class='ovr')
			f1_score = metrics.f1_score(np.argmax(z_test, 1), np.argmax(z_prediction, 1), average = 'weighted')

			print('[ Train Class Loss = {:.4f} Train Acc = {:.4f} ]'.format(avg_class_loss, avg_acc))
			print('[ Test Class Loss = {:.4f}  Test Total Acc : {:.4f}  Test Acc_adc : {:.4f}  Test Acc_lcc : {:.4f}  Test Acc_scc : {:.4f}  Test Auc : {:.4f} Test F1 Score : {:.4f}]\n'.format(float(test_class_loss), float(acc), float(acc_for_each[0]), float(acc_for_each[1]), float(acc_for_each[2]), float(auc), float(f1_score)))

			if acc >= max_total_acc :
				max_epoch = epoch+1
				max_class_loss = test_class_loss
				max_total_acc = acc
				max_acc_adc = acc_for_each[0]
				max_acc_lcc = acc_for_each[1]
				max_acc_scc = acc_for_each[2]
				##
				max_auc = auc
				##
				max_f1_score = f1_score
				max_z_prediction = z_prediction
				max_z_pred_sm = z_pred_sm

			current_class_score.extend([epoch+1, avg_class_loss, avg_acc, test_class_loss, acc,acc_for_each[0], acc_for_each[1], acc_for_each[2], auc, f1_score])
			epoch_class_score.loc[len(epoch_class_score)] = current_class_score
			epoch_class_score.to_csv("./"+ resDirName + filename+"/"+filename+"_epoch_class_score.csv", index=False)


			if test_dice >= max_dice :
				max_epoch = epoch+1
				max_dice = test_dice
				max_iou  = test_iou
				max_seg_loss = test_seg_loss
				max_y_prediction = y_prediction
				max_dice_list = test_dice_list

			current_seg_score.extend([epoch+1, avg_seg_loss, avg_dice, avg_iou, test_seg_loss, test_dice, test_iou])
			#epoch_seg_score.loc[len(epoch_seg_score)] = current_seg_score
			#epoch_seg_score.to_csv("./"+ resDirName+filename+"/"+filename+"_epoch_seg_score.csv", index=False)

	max_dice_img = max(max_dice_list) #max dice img dice
	max_dice_img_idx = max_dice_list.index(max_dice_img) #img index

	score = pd.DataFrame(columns = ['max_epoch', 'max_ae_loss', 'max_seg_loss', 'max_dice', 'max_iou', 'max_total_acc', 'max_acc_adc', 'max_acc_lcc', 'max_acc_scc', 'max_auc', 'max_f1_score', 'max_dice_img_idx', 'max_dice_img'])
	score.loc[len(score)] = [max_epoch, max_ae_loss, max_seg_loss, max_dice, max_iou, max_total_acc, max_acc_adc, max_acc_lcc, max_acc_scc, max_auc, max_f1_score, max_dice_img_idx, max_dice_img]
	score.to_csv("./" + resDirName +filename+"/"+filename+"_score.csv")

	#print("AE LOSS: {:.4f}".format(float(max_ae_loss)))
	print("SEG LOSS: {:.4f}".format(max_seg_loss))
	print("DICE: {:.4f}".format(max_dice))
	print("IOU: {:.4f}".format(max_iou))
	print("TOTAL ACC: {:.4f}".format(max_total_acc))
	print("ACC_adc: {:.4f}".format(max_acc_adc))
	print("ACC_lcc: {:.4f}".format(max_acc_lcc))
	print("ACC_scc: {:.4f}".format(max_acc_scc))
	print("AUC: {:.4f}".format(max_auc))
	print("F1 Score: {:.4f}".format(max_f1_score))
	print(max_dice_img_idx+1, max_dice_img)

	print('Saving prediction numpy...')
	np.save('./'+ resDirName + filename+'/y_test.npy', y_test)
	np.save('./'+ resDirName + filename+'/pred.npy', max_y_prediction)
	
	print('Saving prediction dataframe...')
	z_prediction_raw = pd.DataFrame(max_z_prediction)
	z_pred_sm_df = pd.DataFrame(max_z_pred_sm)
	z_prediction_raw.to_csv('./'+ resDirName + filename+'/'+filename+'_z_prediction_raw.csv')
	z_pred_sm_df.to_csv('./'+ resDirName + filename+'/'+filename+'_z_prediction_sm.csv')

	z_labels = np.argmax(z_test, 1)
	z_preds = np.argmax(max_z_prediction,1)
	z_pred_df = pd.DataFrame(z_labels, columns=['labels'])
	z_pred_df['labels'] = z_pred_df['labels'].round(decimals = 4)
	z_pred_df['prediction'] = z_preds
	z_pred_df.to_csv('./' + resDirName + filename+'/'+filename+'_z_prediction.csv')


	y_test_img = y_test
	print('Finish...')
