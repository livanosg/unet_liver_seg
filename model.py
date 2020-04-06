from tensorflow.compat.v1.initializers import constant, glorot_normal
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, ReLU
from tensorflow.keras.layers import Dropout, Concatenate, BatchNormalization
from tensorflow import name_scope


class Unet:
    def __init__(self, params):
        self.batch_norm = params['batch_norm']
        self.dropout = params['dropout']
        self.classes = params['classes']

    def conv_bn_dr(self, input_tensor, filters):
        conv = Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_initializer=glorot_normal())(input_tensor)
        if self.batch_norm:
            conv = BatchNormalization()(conv)
        conv = ReLU(negative_slope=0.1, )(conv)
        if self.dropout:
            conv = Dropout(rate=self.dropout)(conv)
        return conv

    def convolution_layer(self, input_tensor, filters):
        conv = self.conv_bn_dr(input_tensor=input_tensor, filters=filters)
        conv = self.conv_bn_dr(input_tensor=conv, filters=filters)
        return conv

    def down_layer(self, input_tensor, filters):
        connection = self.convolution_layer(input_tensor=input_tensor, filters=filters)
        output = MaxPooling2D(pool_size=(2, 2), strides=2)(connection)
        return output, connection

    def trans_conv(self, input_tensor, filters):
        up_conv = Conv2DTranspose(filters=filters, kernel_size=2, strides=2, padding='same',
                                  kernel_initializer=glorot_normal(), bias_initializer=constant(0.1),)(input_tensor)
        if self.batch_norm:
            up_conv = BatchNormalization()(up_conv)
        up_conv = ReLU(negative_slope=0.1)(up_conv)
        if self.dropout:
            up_conv = Dropout(self.dropout)(up_conv)
        return up_conv

    def upconv_layer(self, input_tensor, connection, filters):
        trans = self.trans_conv(input_tensor=input_tensor, filters=filters)
        up_conv = Concatenate()([connection, trans])
        return self.convolution_layer(input_tensor=up_conv, filters=filters // 2)

    def model(self, input_tensor):
        with name_scope('Down_1'):
            output, connection_1 = self.down_layer(input_tensor=input_tensor, filters=32)
        with name_scope('Down_2'):
            output, connection_2 = self.down_layer(input_tensor=output, filters=64)
        with name_scope('Down_3'):
            output, connection_3 = self.down_layer(input_tensor=output, filters=128)
        with name_scope('Down_4'):
            output, connection_4 = self.down_layer(input_tensor=output, filters=256)
        with name_scope('Bridge'):
            output = self.convolution_layer(input_tensor=output, filters=512)
        with name_scope('Up_1'):
            output = self.upconv_layer(input_tensor=output, connection=connection_4, filters=256)
        with name_scope('Up_2'):
            output = self.upconv_layer(input_tensor=output, connection=connection_3, filters=128)
        with name_scope('Up_3'):
            output = self.upconv_layer(input_tensor=output, connection=connection_2, filters=64)
        with name_scope('Up_4'):
            output = self.upconv_layer(input_tensor=output, connection=connection_1, filters=32)
        with name_scope('Output'):
            with name_scope('Logits'):
                logits = Conv2D(filters=self.classes, kernel_size=1, kernel_initializer=glorot_normal(), padding='same')(output)
        return logits
