import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.merge import add
from keras import regularizers



def resnet8(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.
    
    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.
       
    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(img_input)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = keras.layers.normalization.BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x2 = keras.layers.normalization.BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])


    # Second residual block
    x4 = keras.layers.normalization.BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x4 = keras.layers.normalization.BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = keras.layers.normalization.BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x6 = keras.layers.normalization.BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=regularizers.l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Steering channel, first head
    steer = Dense(output_dim)(x)

    # Collision channel, second head
    coll = Dense(output_dim)(x)
    coll = Activation('sigmoid')(coll)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[steer, coll])
    print(model.summary())

    return model

model = resnet8(224,224,3,1)
print(model.output_shape)

# below is the model summary
'''
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 224, 224, 3)  0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 112, 112, 32) 2432        input_1[0][0]                    
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 55, 55, 32)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
batch_normalization_1 (BatchNor (None, 55, 55, 32)   128         max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 55, 55, 32)   0           batch_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 28, 28, 32)   9248        activation_1[0][0]               
__________________________________________________________________________________________________
batch_normalization_2 (BatchNor (None, 28, 28, 32)   128         conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 28, 28, 32)   0           batch_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 28, 28, 32)   1056        max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 28, 28, 32)   9248        activation_2[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 28, 28, 32)   0           conv2d_4[0][0]                   
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
batch_normalization_3 (BatchNor (None, 28, 28, 32)   128         add_1[0][0]                      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 28, 28, 32)   0           batch_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 14, 14, 64)   18496       activation_3[0][0]               
__________________________________________________________________________________________________
batch_normalization_4 (BatchNor (None, 14, 14, 64)   256         conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 14, 14, 64)   0           batch_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 14, 14, 64)   2112        add_1[0][0]                      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 14, 14, 64)   36928       activation_4[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 14, 14, 64)   0           conv2d_7[0][0]                   
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
batch_normalization_5 (BatchNor (None, 14, 14, 64)   256         add_2[0][0]                      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 14, 14, 64)   0           batch_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 7, 7, 128)    73856       activation_5[0][0]               
__________________________________________________________________________________________________
batch_normalization_6 (BatchNor (None, 7, 7, 128)    512         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 7, 7, 128)    0           batch_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 7, 7, 128)    8320        add_2[0][0]                      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 7, 7, 128)    147584      activation_6[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 7, 7, 128)    0           conv2d_10[0][0]                  
                                                                 conv2d_9[0][0]                   
__________________________________________________________________________________________________
flatten_1 (Flatten)             (None, 6272)         0           add_3[0][0]                      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 6272)         0           flatten_1[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 6272)         0           activation_7[0][0]               
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           62730       dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           62730       dropout_1[0][0]                  
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 10)           0           dense_2[0][0]                    
==================================================================================================
Total params: 436,148
Trainable params: 435,444
Non-trainable params: 704
__________________________________________________________________________________________________

'''