import os
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
x = Dropout(0.5)(x)
output = Dense(4, activation='softmax')(x)  # 3 classes

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Data generators
train_path = './output_dataset/train'
val_path = './output_dataset/val'

train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_path, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_path, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Train model
checkpoint = ModelCheckpoint('healthy_vs_rotten.h5', save_best_only=True)
earlystop = EarlyStopping(patience=3)

model.fit(train_data, epochs=10, validation_data=val_data, callbacks=[checkpoint, earlystop])

model.save("vgg16_model.keras")

