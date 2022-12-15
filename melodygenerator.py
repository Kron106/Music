import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="D:\code\Python code\pythonProject\generating-melodies-with-rnn-lstm\9 - Converting Generated Melodies to MIDI\model.h5"):
        """Constructor that initialises TensorFlow model"""
        #加载模型
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)
        #加载映射
        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)
        #添加开始符号
        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.
        :return melody (list of str): List with symbols representing a melody
        """
        #
        #Create seed with start symbols
        #创建一个旋律
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # 将映射转换为int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # 限制种子为最大序列长度
            seed = seed[-max_sequence_length:]

            # 独热编码
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            onehot_seed = onehot_seed[np.newaxis, ...]

            # 进行预测
            probabilities = self.model.predict(onehot_seed)[0]
            # [0.1, 0.2, 0.1, 0.6] -> 1
            output_int = self._sample_with_temperature(probabilities, temperature)

            # 更新种子
            seed.append(output_int)

            # 将输出转换为符号
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # 处理结束符号
            if output_symbol == "/":
                break

            # 更新旋律
            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """将旋律转换为 MIDI 文件
        :return:
        """

        # create a music21 stream 充当容器
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # 解析旋律中的所有符号并创建 Note/rest 对象
        for i, symbol in enumerate(melody):

            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed2, 500, SEQUENCE_LENGTH, 1)
    print(melody)
    mg.save_melody(melody)




















