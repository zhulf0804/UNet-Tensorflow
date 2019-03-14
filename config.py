import tensorflow as tf

flags = tf.app.flags
FlAGS = flags.FLAGS

# about dataset information

flags.DEFINE_string('data_path', './dataset/my_dataset', 'the directory of the datasets')
flags.DEFINE_string('img_dir_name', 'images', 'the directory of the raw images')
flags.DEFINE_string('annotation_dir_name', 'annosRaw', 'the directory of the annotation images')
flags.DEFINE_string('train_list_file', 'train.txt', 'the .txt file to save the training images filename') # one line one name, not including the suffix name
flags.DEFINE_string('trainval_list_file', 'trainval.txt', 'the .txt file to save the training and val images filename') # one line one name, not including the suffix name
flags.DEFINE_string('suffix_name', '.png', 'the suffix name of training images') # one line one name, not including the suffix name



# network parameter
flags.DEFINE_integer('batch_size', 4, 'the training batch size')
flags.DEFINE_integer('img_size', 512, 'the size of input images')
flags.DEFINE_integer('num_channels', 3, 'the channel of the input images')
flags.DEFINE_integer('classes', 4, 'the number of the classes, including the background')

# hyper parameter
flags.DEFINE_float('learning_rate', 1e-4, 'the init learning rate')
flags.DEFINE_integer('max_steps', 30000, 'the init learning rate')
flags.DEFINE_string('weighted', "no", 'choose the model') # if weighted is yes, you need to config the loss_weight in the train.py file.



# about dataset information
data_path = FlAGS.data_path
img_dir_name = FlAGS.img_dir_name
annotation_dir_name = FlAGS.annotation_dir_name
train_list_file = FlAGS.train_list_file
trainval_list_file = FlAGS.trainval_list_file
suffix_name = FlAGS.suffix_name

# network parameter
batch_size = FlAGS.batch_size
img_size = FlAGS.img_size
num_channels = FlAGS.num_channels
classes = FlAGS.classes

# hyper parameter
learning_rate = FlAGS.learning_rate
max_steps = FlAGS.max_steps
weighted = FlAGS.weighted


