from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense
from keras.models import Model
import os
from keras.datasets import cifar10
from keras.utils import to_categorical

def residual_block(x, filters, kernel_size=3, stride=1):
    # 主分支
    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # 主分支的第二层
    y = Conv2D(filters, kernel_size=kernel_size, padding='same')(y)
    y = BatchNormalization()(y)

    # 跳跃连接
    if stride != 1 or x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(x)

    # 相加
    y = Add()([x, y])
    y = Activation('relu')(y)
    return y

def build_resnet(input_shape=(32, 32, 3), num_classes=10, num_filters=16, depth=50, width=50):
    # 输入层
    input_layer = Input(shape=input_shape)

    # 初始卷积层
    x = Conv2D(num_filters, kernel_size=3, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 堆叠残差块
    for i in range(depth):
        for j in range(width):
            stride = 2 if j == 0 and i != 0 else 1
            x = residual_block(x, filters=num_filters, stride=stride)

        num_filters *= 2  # 每个阶段加倍滤波器数目

    # 全局平均池化
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    # 构建模型
    model = Model(inputs=input_layer, outputs=output_layer, name='resnet')
    return model




(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


# 构建深度为50，宽度为50的 ResNet 模型
resnet_model = build_resnet(depth=50, width=50)

# 编译模型
resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
resnet_model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

resnet_model.summary()