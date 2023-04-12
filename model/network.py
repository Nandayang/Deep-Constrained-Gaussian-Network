from tensorflow import keras
from config import config

def create_unet_model():
    mobilenet_model = keras.applications.MobileNetV2(input_shape=(config.img_size, config.img_size, 3),
                                                     include_top=False, weights='imagenet')

    block_1 = mobilenet_model.get_layer('block_1_expand_relu').output
    block_3 = mobilenet_model.get_layer('block_3_expand_relu').output
    block_6 = mobilenet_model.get_layer('block_6_expand_relu').output
    block_13 = mobilenet_model.get_layer('block_13_expand_relu').output
    block_16 = mobilenet_model.get_layer('block_16_expand_relu').output

    up_16 = UpSampleBlock(256)(block_16)
    merge_16 = keras.layers.Concatenate()([up_16, block_13])

    up_17 = UpSampleBlock(128)(merge_16)
    merge_17 = keras.layers.Concatenate()([up_17, block_6])

    up_18 = UpSampleBlock(64)(merge_17)
    merge_18 = keras.layers.Concatenate()([up_18, block_3])

    up_19 = UpSampleBlock(32)(merge_18)
    merge_19 = keras.layers.Concatenate()([up_19, block_1])

    up_20 = UpSampleBlock(config.ClusterNo)(merge_19)
    output = keras.activations.softmax(up_20)

    model = keras.Model(inputs=mobilenet_model.input, outputs=output)
    return model


def UpSampleBlock(filters):
    def unsampleblock(x):
        x = keras.layers.UpSampling2D()(x)
        x = keras.layers.Conv2D(filters, 3, kernel_initializer='he_normal', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        return x
    return unsampleblock



def create_unet_model_supervised():
    mobilenet_model = keras.applications.MobileNetV2(input_shape=(config.img_size, config.img_size, 3),
                                                     include_top=False, weights='imagenet')

    block_1 = mobilenet_model.get_layer('block_1_expand_relu').output
    block_3 = mobilenet_model.get_layer('block_3_expand_relu').output
    block_6 = mobilenet_model.get_layer('block_6_expand_relu').output
    block_13 = mobilenet_model.get_layer('block_13_expand_relu').output
    block_16 = mobilenet_model.get_layer('block_16_expand_relu').output

    up_16 = UpSampleBlock(256)(block_16)
    merge_16 = keras.layers.Concatenate()([up_16, block_13])

    up_17 = UpSampleBlock(128)(merge_16)
    merge_17 = keras.layers.Concatenate()([up_17, block_6])

    up_18 = UpSampleBlock(64)(merge_17)
    merge_18 = keras.layers.Concatenate()([up_18, block_3])

    up_19 = UpSampleBlock(32)(merge_18)
    merge_19 = keras.layers.Concatenate()([up_19, block_1])

    up_20 = UpSampleBlock(1)(merge_19)
    output = keras.activations.sigmoid(up_20)

    model = keras.Model(inputs=mobilenet_model.input, outputs=output)
    return model

def create_biunet_model():
    mobilenet_model = keras.applications.MobileNetV2(input_shape=(config.img_size, config.img_size, 3),
                                                     include_top=False, weights='imagenet')

    block_1 = mobilenet_model.get_layer('block_1_expand_relu').output
    block_3 = mobilenet_model.get_layer('block_3_expand_relu').output
    block_6 = mobilenet_model.get_layer('block_6_expand_relu').output
    block_13 = mobilenet_model.get_layer('block_13_expand_relu').output
    block_16 = mobilenet_model.get_layer('block_16_expand_relu').output

    up_16 = UpSampleBlock(256)(block_16)
    merge_16 = keras.layers.Concatenate()([up_16, block_13])

    up_17 = UpSampleBlock(128)(merge_16)
    merge_17 = keras.layers.Concatenate()([up_17, block_6])

    up_18 = UpSampleBlock(64)(merge_17)
    merge_18 = keras.layers.Concatenate()([up_18, block_3])

    up_19 = UpSampleBlock(32)(merge_18)
    merge_19 = keras.layers.Concatenate()([up_19, block_1])

    up_20 = UpSampleBlock(config.ClusterNo)(merge_19)
    output = keras.activations.softmax(up_20)

    model = keras.Model(inputs=mobilenet_model.input, outputs=output)
    return model



if __name__ == "__main__":
    models = create_unet_model()