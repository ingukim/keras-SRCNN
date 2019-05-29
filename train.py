import DataLoader
import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

train_path='D:/SRCNN/SRCNN_KERAS/patch/train.h5'
val_path='D:/SRCNN/SRCNN_KERAS/patch/val.h5'

data, label=DataLoader.prepare_data(train_path)
val_data, val_label=DataLoader.prepare_data(val_path)

#Normalization
data=data.astype(float) / 255.
label=label.astype(float) / 255.
val_data=val_data.astype(float) / 255.
val_label=val_label.astype(float) / 255.

srcnn_model = Model.model()
early_stopping=EarlyStopping(monitor='val_PSNR', patience=20, mode='max')

file_path='./checkpoints/weights-improvement-{epoch:02d}-{val_PSNR:0.3f}-{val_loss:0.2f}.h5'

checkpoint = ModelCheckpoint(file_path, monitor='val_PSNR', verbose=1, save_best_only=True, mode='max')

hist=srcnn_model.fit(data, label, epochs=10, batch_size=32, validation_data=(val_data, val_label), callbacks=[early_stopping, checkpoint])

'''
fig, loss_ax=plt.subplots()
acc_ax=loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label = 'train_loss')
loss_ax.plot(hist.history['val_loss'],'r',label='val_loss')
    
acc_ax.plot(hist.history['PSNR'],'b',label='train_PSNR')
acc_ax.plot(hist.history['val_PSNR'], 'g', label='val_PSNR')
    
oss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('PSNR')
    
loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')
    
plt.show()
'''
