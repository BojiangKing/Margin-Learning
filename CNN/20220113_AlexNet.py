#%%
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from contextlib import ExitStack 
#%%
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data() 

X_valid, X_train = X_train_full[:5000], X_train_full[5000:] 
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# 打乱训练集次序
train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train))
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
del X_train_full, y_train_full, X_test, y_test
del X_valid, X_train, y_valid, y_train
#%%
# 数据集过于庞大，所以我们有必要将数据存到文件中，即调即用
BytesList = tf.train.BytesList 
FloatList = tf.train.FloatList 
Int64List = tf.train.Int64List 
Feature = tf.train.Feature 
Features = tf.train.Features 
Example = tf.train.Example 
def create_example(image, label):     
    image_data = tf.io.serialize_tensor(image)     
    # image_data = tf.io.encode_jpeg(image[..., np.newaxis])     
    return Example(         
        features=Features(             
            feature={                 
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),                 
                "label": Feature(int64_list=Int64List(value=[label])),             
            })) 
def write_tfrecords(name, dataset, n_shards=10):     
    paths = ["{}.tfrecord-{:05d}-of-{:05d}".format(name, index, n_shards)              
            for index in range(n_shards)]     
    with ExitStack() as stack:         
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))                    
                    for path in paths]         
        for index, (image, label) in dataset.enumerate():             
            shard = index % n_shards             
            example = create_example(image, label)             
            writers[shard].write(example.SerializeToString())     
    return paths 
train_filepaths = write_tfrecords("my_fashion_mnist.train", train_set) 
valid_filepaths = write_tfrecords("my_fashion_mnist.valid", valid_set) 
test_filepaths = write_tfrecords("my_fashion_mnist.test", test_set)

del train_set, valid_set, test_set
#%%
def preprocess(tfrecord): 
    feature_descriptions = { 
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""), 
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1) 
    } 
    example = tf.io.parse_single_example(tfrecord, feature_descriptions) 
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8) 
    #image = tf.io.decode_jpeg(example["image"]) 
    image = tf.reshape(image, shape=[28, 28, 1]) 
    image = tf.image.resize(image, [224,224]) # if we want to resize 
    return image, tf.one_hot(example["label"], depth=10)
def mnist_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None, n_parse_threads=5, batch_size=32, cache=True): 
    dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=n_read_threads) 
    if cache: 
        # The first time the dataset is iterated over, 
        # its elements will be cached either in the specified file or in memory. 
        # Subsequent iterations will use the cached data. 
        dataset = dataset.cache() 
    if shuffle_buffer_size: 
        dataset = dataset.shuffle(shuffle_buffer_size) 
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads) 
    dataset = dataset.batch(batch_size) 
    return dataset.prefetch(1) 
train_set = mnist_dataset(train_filepaths, shuffle_buffer_size=60000) 
valid_set = mnist_dataset(valid_filepaths) 
test_set = mnist_dataset(test_filepaths)
# %%
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
        # 这里，我们使用一个11*11的更大窗口来捕捉对象。
        # 同时，步幅为4，以减少输出的高度和宽度。
        # 另外，输出通道的数目远大于LeNet
        keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu', input_shape=[224, 224, 1]),
        keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
        keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        keras.layers.MaxPool2D(pool_size=3, strides=2),
        # 使用三个连续的卷积层和较小的卷积窗口。
        # 除了最后的卷积层，输出通道的数量进一步增加。
        # 在前两个卷积层之后，汇聚层不用于减少输入的高度和宽度
        keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        keras.layers.MaxPool2D(pool_size=3, strides=2),
        keras.layers.Flatten(),
        # 这里，全连接层的输出数量是LeNet中的好几倍。使用dropout层来减轻过拟合
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        # 最后是输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
        keras.layers.Dense(10)
    ])

model.summary()
model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["accuracy"]) 
# %%
model.fit(train_set, epochs=5, validation_data=valid_set)
# %%
