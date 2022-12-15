import tensorflow.keras as keras
from preprocess import generate_training_sequences, SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 2
BATCH_SIZE = 64
SAVE_MODEL_PATH = "model.h5"


def build_model(output_units, num_units, loss, learning_rate):
    """Builds and compiles model

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply

    :return model (tf model): Where the magic happens :D
    """

    # 创建模型架构
    input = keras.layers.Input(shape=(None, output_units))
    x = keras.layers.LSTM(num_units[0])(input)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units, activation="softmax")(x)

    model = keras.Model(input, output)

    # 编译模型
    model.compile(loss=loss,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])

    model.summary()

    return model


def train(output_units=OUTPUT_UNITS, num_units=NUM_UNITS, loss=LOSS, learning_rate=LEARNING_RATE):
    """训练并保存模型.

    :param output_units (int): Num output units
    :param num_units (list of int): Num of units in hidden layers
    :param loss (str): Type of loss function to use
    :param learning_rate (float): Learning rate to apply
    """

    # 生成训练序列
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    # 构建网络
    model = build_model(output_units, num_units, loss, learning_rate)

    # 训练模型
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # 保存模型
    model.save(SAVE_MODEL_PATH)


if __name__ == "__main__":
    train()