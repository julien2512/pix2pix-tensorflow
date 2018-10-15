from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
#from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")

parser.add_argument("--f1", type=int, default=40, help="number of full layer to get the commands")
parser.add_argument("--f2", type=int, default=40, help="number of full layer to get the next bet")
parser.add_argument("--frames", type=int, default=5, help="number of frames for each to generate commands")
parser.add_argument("--command_fire_level", type=float, default=0.003, help="level at which a command is fired")
parser.add_argument("--magic_weight", type=float, default=0.005, help="weight of magic formula over L1 loss") # 0.009
parser.add_argument("--l1_weight", type=float, default=0.005, help="weight of L1 loss")  # 0.001
parser.add_argument("--perf_weight", type=float, default=0.01, help="weight of perf loss")
parser.add_argument("--nodrop", type=bool, default=True, help="Desactivate the drop layer")

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple("Examples", "paths, inputs, meta, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss, gen_grads_and_vars, situation, train, next_commands")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def situation(input,situation_size):
    with tf.variable_scope("situation"):
        situation = tf.layers.dense(input,situation_size,activation=tf.nn.sigmoid)
        return situation

def commands(input, out_channels):
    with tf.variable_scope("commands"):
        commands = tf.layers.dense(input,out_channels); 
        return commands

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def deconv(batch_input, out_channels):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        raise Exception("all image files names are not numbers")

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])
        raw_input = preprocess(raw_input)

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = a.seed
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(raw_input)

    # Adding meta & targets inputs
    meta_paths = glob.glob(os.path.join(a.input_dir, "*.meta"))
    if len(meta_paths) == 0:
        raise Exception("input_dir contains no meta files")
    if len(meta_paths) != len(input_paths):
        raise Exception("input_dir contains not enough meta files")

    if all(get_name(path).isdigit() for path in meta_paths):
        meta_paths = sorted(meta_paths, key=lambda path: int(get_name(path)))
    else:
        raise Exception("all meta files names are not numbers")

    # prepare data from meta file
    def split(contents):
        tensor = tf.string_split([contents],'\t').values
        tensor = tf.reshape(tensor,[12])
        tensor = tf.string_to_number(tensor,tf.float32)
        return tensor

    with tf.name_scope("load_meta"):
        meta_path_queue = tf.train.string_input_producer(meta_paths)
        meta_reader = tf.WholeFileReader()
        meta_paths, meta_contents = meta_reader.read(meta_path_queue)
#        meta_contents = tf.Print(meta_contents, [meta_contents], "meta_contents:", summarize=100)
        meta_input = split(meta_contents)

    target_paths = glob.glob(os.path.join(a.input_dir, "*.meta_targets"))
#    print(target_paths)
    if len(target_paths) == 0:
        target_input = None
    else:
        if all(get_name(path).isdigit() for path in target_paths):
            target_paths = sorted(target_paths, key=lambda path: int(get_name(path)))
        else:
            raise Exception("all meta targets files names are not numbers")

        with tf.name_scope("load_targets"):
            targets_path_queue = tf.train.string_input_producer(target_paths)
            targets_reader = tf.WholeFileReader()
            target_paths, target_contents = targets_reader.read(targets_path_queue)
#            target_contents = tf.Print(target_contents, [target_contents], "target_contents:", summarize=100)
            target_input = split(target_contents)

#    meta_input = tf.Print(meta_input, [meta_input], "load meta_input:",summarize=100)
#    if target_input is not None:
#      target_input = tf.Print(target_input, [target_input], "load targets_input:", summarize=100)

    if target_input is not None:
      paths_batch, inputs_batch, meta_batch, targets_batch = tf.train.batch([paths, input_images, meta_input, target_input], batch_size=a.batch_size)
    else:
      paths_batch, inputs_batch, meta_batch = tf.train.batch([paths, input_images, meta_input], batch_size=a.batch_size)
      targets_batch = None

    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        meta=meta_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_situation_analysis_generator(generator_inputs):
    layers = []

#    generator_inputs = tf.Print(generator_inputs,[generator_inputs],"generator_inputs:",summarize=256)

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = lrelu(generator_inputs, 0.2)
        output = conv(output, a.ngf, stride=2)
#        output = tf.Print(output,[output],"output encoder_1",summarize=100)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
#        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            output = batchnorm(convolved)
#            output = tf.Print(output,[output],"output encoder_%d" % (len(layers)+1),summarize=100)
            layers.append(output)

#    with tf.variable_scope("situation"):
#        situation = situation(layers[-1], situation_size)

    # the last layer to get the situation analysis
    # aims to reduce the shape because of the encode 8 zero bug
    with tf.variable_scope("situation"):
        # fully_connected: [batch, 2, 2, 8*ngf] => [batch, 2*4*ngf]
        rectified = tf.nn.relu(layers[-1])
        output = tf.reshape(rectified,[a.batch_size,2*2*a.ngf*8])
        output = commands(output, 2*a.ngf*4)
        output = tf.Print(output,[output],"situation",summarize=1000)
        layers.append(output)
    # to train computer to cheat is not allowed
    # so that meta_inputs is not really necessary in the model
    #    meta_inputs = tf.reshape(meta_inputs,[-1,4])
    #    layers.append(tf.concat(situation_analysis,meta_inputs))
#        situation_analysis = tf.Print(situation_analysis,[situation_analysis],'situation_analysis:',summarize=100)

    return layers[-1]

def create_next_commands_generator(situation_analysis):
    layers = []

    layers.append(situation_analysis)

    for next_commands_channels in range(1,a.f1-1):
        # fully_connected: [batch, 4*ngf*2] => [batch, 4*ngf*2]
        with tf.variable_scope("f1_fully_connected_%d" % (next_commands_channels)):
            rectified = tf.nn.relu(layers[-1])
            output = commands(rectified, a.ngf*8)
            if not a.nodrop and next_commands_channels<a.f1-5:
                 output = tf.nn.dropout(output, keep_prob=0.5, seed=a.seed)
#            output = tf.Print(output,[output],"output f1_%d" % (next_commands_channels),summarize=100)
            layers.append(output)

    with tf.variable_scope("next_commands"):
        # fully_connected: [batch, 4 * ngf * 2] => [batch, frames, 12]
        rectified = tf.nn.relu(layers[-1])
        rectified = tf.reshape(rectified, [a.batch_size,4*a.ngf*2])
        output = commands(rectified, 12*a.frames) # 10 sets of 12 commands
        output = tf.reshape(output, [a.batch_size,a.frames,12])
        output = tf.abs(tf.tanh(output)); # probability values for the commands, integers later for the bets
        output = tf.Print(output,[output],"output next_commands",summarize=100)
        layers.append(output)

    return layers[-1]

def create_next_bet_generator(situation_analysis, next_commands):
    layers = []

    # commands & situation analysis
    input = tf.concat((tf.reshape(next_commands,[a.batch_size,3*a.frames,4]), tf.reshape(situation_analysis,[a.batch_size,2*a.ngf,4])), axis=1)
    layers.append(input)

    with tf.variable_scope("f2_fully_connected_1"):
        rectified = tf.nn.relu(layers[-1])
        output = tf.reshape(rectified,[a.batch_size,2*a.ngf*4+12*a.frames])
        output = commands(output, a.ngf*8+12*a.frames)
        layers.append(output)

    for next_bet_channels in range(2,a.f2-1):
        # fully_connected: [batch, 12*11+ngf*8] => [batch, 12*frames+ngf*8]
        with tf.variable_scope("f2_fully_connected_%d" % (next_bet_channels)):
            rectified = tf.nn.relu(layers[-1])
            output = commands(rectified, a.ngf*8+12*a.frames)
            if not a.nodrop and next_bet_channels<a.f2-5:
                 output = tf.nn.dropout(output, keep_prob=0.5, seed=a.seed)
#            output = tf.Print(output,[output],"output f2_%d" % (next_bet_channels), summarize=100)
            layers.append(output)

    # next bet [batch, 12]
    with tf.variable_scope("next_bet"):
        rectified = tf.nn.relu(layers[-1])
        output = commands(rectified, 12)
        output = tf.abs(output)
        output = tf.scalar_mul(1000,output) # 1000 is totally arbitrary
#        output = tf.Print(output,[output],"output next_bet", summarize=100)
        layers.append(output)

    return layers[-1]

    # [batch,12]+[batch,frames,12] => [batch,frames+1,12]
#    concat = tf.concat((tf.reshape(next_bet,[a.batch_size,1,12]),next_commands),axis=1)
#    concat = tf.Print(concat,[concat],'generator_concat:',summarize=100)
#    return concat


# assume bet[0] = P1 life
#        bet[1] = P2 life
#        bet[2] = time
#
# this magic formula aims to 
# optimize p1_life & p2_life by time
def get_situation_loss(actual, bet):

#    p1_life = bet[:,0:1]           # ex:  104/176   196/176
#    p2_life = bet[:,1:2]           # ex:  176/176   655/176
#    actual_time    = actual[:,2:3] # ex:  107/147   130/147
#    actual_p1_life = actual[:,0:1] # ex:             26/176
#    actual_p2_life = actual[:,1:2] # ex:  90/176    176/176
#    time = tf.Print(time,[time],"time:")

#    max_kill_time = tf.fill([a.batch_size,1],100.0)
#    max_time = tf.fill([a.batch_size,1],153.0)
#    max_life = tf.fill([a.batch_size,1],176.0)

#    actual = actual[:,0:3]
#    bet    = bet[:,0:3]

    actual = tf.slice(actual,[0,0],[a.batch_size,3])
    bet    = tf.slice(bet,[0,0],[a.batch_size,3])

    return  tf.abs(actual-bet)

# assume bet[0] = P1 life
#        bet[1] = P2 life
#        bet[2] = time
#
# this magic formula aims to 
# optimize p1_life & p2_life by time
def get_magic_target(actual, bet):

    p1_life = bet[:,0:1]           # ex:  104/176   196/176
    p2_life = bet[:,1:2]           # ex:  176/176   655/176
    actual_time    = actual[:,2:3] # ex:  107/147   130/147
    actual_p1_life = actual[:,0:1] # ex:             26/176
    actual_p2_life = actual[:,1:2] # ex:  90/176    176/176
#    time = tf.Print(time,[time],"time:")

    max_kill_time = tf.fill([a.batch_size,1],100.0)
    max_time = tf.fill([a.batch_size,1],153.0)
    max_life = tf.fill([a.batch_size,1],176.0)

    magic_p2_life = actual_time                              # 107                  130
    magic_p2_life = tf.multiply(magic_p2_life,max_life)      # 107*176 = 18832      130*176 = 22880
    magic_p2_life = tf.divide(magic_p2_life, max_time)       # 18832 / 147 = 128    22880 / 147 = 155
    magic_p2_life = tf.minimum(actual_p2_life,magic_p2_life) # min(90,176) = 90     min(176,155) = 155
    
    magic_p1_life = actual_time                                # 107                  130
    magic_p1_life = tf.multiply(magic_p1_life,max_life)        # 107*176 = 18832      130*176 = 22880
    magic_p1_life = tf.divide(magic_p1_life, max_time)         # 18832 / 147 = 128    22880 / 147 = 155
    magic_p1_life = tf.minimum(max_life,tf.maximum(actual_p1_life,magic_p1_life)) # min(176,max(104,128)) = 128  min(176,max(26,155)) = 155

    magic_bet = tf.concat((magic_p1_life,magic_p2_life,actual_time,tf.fill([a.batch_size,12-3],0.0)),axis=1)

    return magic_bet

# how good Ryu is
def get_performance(last, actual, bet):

    last_p1_life = tf.reshape(last[0][0:1],[])
    last_p2_life = tf.reshape(last[0][1:2],[])
    p1_life = tf.reshape(actual[0][0:1],[])           # ex:  104/176   196/176
    p2_life = tf.reshape(actual[0][1:2],[])           # ex:  176/176   655/176
    actual_time = tf.reshape(actual[0][2:3],[]) # ex:  107/147   130/147
    magic_bet = get_magic_target(actual, bet)
    magic_p1_life = tf.reshape(magic_bet[0][0:1],[])
    magic_p2_life = tf.reshape(magic_bet[0][1:2],[])

    zero = tf.constant(0.0)
    zerol = lambda: zero
    defaut = tf.maximum(zero,-magic_p1_life+p1_life)+tf.maximum(zero,-magic_p2_life+p2_life)
    defaut = tf.Print(defaut,[defaut],"perf_loss:")
    defautl = lambda: defaut

# Ryu hits Zangief => good
# Ryu has hit Zangief enough => good
# Ryu was not hit too much => good
# otherwise, reduce_mean
    perf = tf.case([( tf.greater(last_p2_life, p2_life) , zerol ),  
                    ( tf.greater(magic_p2_life, p2_life), zerol ),  
                    ( tf.greater(p1_life, magic_p1_life), zerol ),] ,
                   default=defautl ) 

    return perf

def create_model(inputs, meta, targets):
    with tf.variable_scope("situation_analysis") as scope:
        situation_analysis = create_situation_analysis_generator(inputs)

    with tf.variable_scope("generator") as scope:
        next_commands = create_next_commands_generator(situation_analysis)
        next_bet = create_next_bet_generator(situation_analysis, next_commands)
        next_bet = tf.reshape(next_bet,[a.batch_size,12])

        #outputs = create_generator(inputs)
#        outputs = create_generator(inputs, meta)  # cheating
#        outputs = tf.Print(outputs,[outputs],"outputs full:",summarize=100)

        #next_commands = outputs[:,1:(a.frames+1)]
        #next_bet = tf.reshape(outputs[:,0:1],[a.batch_size,12])
#        next_commands = tf.Print(next_commands,[next_commands],"next_commands:",summarize=100)
        next_bet = tf.Print(next_bet,[next_bet],"next_bet:",summarize=10)

        situation_targets = targets

    if targets is None:
        targets = next_bet # without targets, no change on model
        situation_targets = situation_analysis

#    with tf.name_scope("generator_loss"):
        # abs(targets - outputs) => 0
#        targets = tf.Print(targets,[targets],"targets:",summarize=100)
#        gen_loss_L1 = tf.reduce_mean(tf.abs(targets[:,0:2] - next_bet[:,0:2])) # back to thirds only
#        gen_loss_L1 = tf.Print(gen_loss_L1,[gen_loss_L1],"gen_loss_L1:")

#    with tf.name_scope("perf"):
#        perf_loss = get_performance(meta, targets, next_bet)
#        perf_loss = tf.Print(perf_loss, [perf_loss], "perf_loss:")

    with tf.name_scope("situation"):
        situation_targets = tf.Print(situation_targets, [situation_targets], "targets:")
        situation = get_situation_loss(situation_targets,situation_analysis)

    with tf.name_scope("p1_life_train"):
        p1_life_loss = 0.1*tf.reduce_mean(tf.slice(situation,[0,0],[a.batch_size,1]))
        p1_life_loss = tf.Print(p1_life_loss,[p1_life_loss],"p1_life_loss:")
        p1_life_tvars = [var for var in tf.trainable_variables() if var.name.startswith("situation")]
        p1_life_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        p1_life_grads_and_vars = p1_life_optim.compute_gradients(p1_life_loss, var_list=p1_life_tvars)
        p1_life_train = p1_life_optim.apply_gradients(p1_life_grads_and_vars)

    with tf.name_scope("p2_life_train"):
        with tf.control_dependencies([p1_life_train]):
            p2_life_loss = 0.1*tf.reduce_mean(tf.slice(situation,[0,1],[a.batch_size,1]))
            p2_life_loss = tf.Print(p2_life_loss,[p2_life_loss],"p2_life_loss:")
            p2_life_tvars = [var for var in tf.trainable_variables() if var.name.startswith("situation")]
            p2_life_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            p2_life_grads_and_vars = p2_life_optim.compute_gradients(p2_life_loss, var_list=p2_life_tvars)
            p2_life_train = p2_life_optim.apply_gradients(p2_life_grads_and_vars)

    with tf.name_scope("time_train"):
        with tf.control_dependencies([p1_life_train,p2_life_train]):
            time_loss = 0.1*tf.reduce_mean(tf.slice(situation,[0,2],[a.batch_size,1]))
            time_loss = tf.Print(time_loss,[time_loss],"time_loss:")
            time_tvars = [var for var in tf.trainable_variables() if var.name.startswith("situation")]
            time_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            time_grads_and_vars = time_optim.compute_gradients(time_loss, var_list=time_tvars)
            time_train = time_optim.apply_gradients(time_grads_and_vars)

    with tf.name_scope("situation_loss"):
        situation_loss = p1_life_loss + p2_life_loss + time_loss
        situation_loss = tf.Print(situation_loss, [situation_loss], "situation_loss:")

#        zero = tf.constant(0.0)
#        situation_loss = tf.case([(tf.equal(perf_loss,zero),lambda:tf.scalar_mul(0.0000001,situation_loss))], default=lambda:situation_loss)
#        situation_loss = tf.Print(situation_loss,[situation_loss],"situation_loss:")


#    with tf.name_scope("magic"):
#        zero = tf.constant(0.0)

#        magic_target = get_magic_target(targets,next_bet)
#        magic_target = tf.Print(magic_target,[magic_target],"magic_target:",summarize=100)
#        magic_loss = tf.reduce_mean(tf.abs(magic_target[:,0:2] - next_bet[:,0:2]))
#        magic_loss = tf.scalar_mul(a.magic_weight, magic_loss)
#        magic_loss = tf.case([(tf.equal(perf_loss, zero), lambda:tf.scalar_mul(0.0000001,magic_loss))], default=lambda:magic_loss)
#        magic_loss  = tf.Print(magic_loss, [magic_loss], "magic_loss:")

#    with tf.name_scope("gen_loss"):
#        zero = tf.constant(0.0)

#        gen_loss = tf.scalar_mul(a.l1_weight,gen_loss_L1)
#        gen_loss = tf.case([(tf.equal(perf_loss,zero),lambda:tf.scalar_mul(0.0000001,gen_loss))], default=lambda:gen_loss)
#        gen_loss = tf.Print(gen_loss,[gen_loss],"gen_loss:")

#    with tf.name_scope("situation_train"):
#        situation_tvars = [var for var in tf.trainable_variables() if var.name.startswith("situation_analysis")]
#        situation_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#        situation_grads_and_vars = situation_optim.compute_gradients(situation_loss, var_list=situation_tvars)
#        situation_train = situation_optim.apply_gradients(situation_grads_and_vars)

#    with tf.name_scope("magic_train"):
#        magic_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#        magic_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#        magic_grads_and_vars = magic_optim.compute_gradients(magic_loss, var_list=magic_tvars)
#        magic_train = magic_optim.apply_gradients(magic_grads_and_vars)

#   
#    with tf.name_scope("generator_train"):
#      with tf.control_dependencies([magic_train]):
#        gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
#        gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
#        gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
#        gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

#    ema = tf.train.ExponentialMovingAverage(decay=0.99)
#    update_losses = ema.apply([gen_loss])

    global_step = tf.contrib.framework.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

#    gen_loss = ema.average(gen_loss)
#    train = tf.group(update_losses, incr_global_step, gen_train)
#    train = tf.group(incr_global_step, gen_train)

# train situation analysis only
    train = tf.group(incr_global_step, time_train)

    return Model(
        gen_loss=situation_loss,
        gen_grads_and_vars=time_grads_and_vars,
        situation=situation_analysis,
        outputs=next_bet,
        next_commands=next_commands,
        train=train,
    )

def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "meta", "targets", "next_commands", "situation"]:
            if kind == "inputs":
              filename = name + ".png"
            else:
              filename = name + "."+kind
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            if len(fetches[kind]) != 0:
                contents = fetches[kind][i]
#                if not kind =="inputs":
#                  print(contents)
                with open(out_path, "wb") as f:
                    f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if tf.__version__ != "1.0.0":
        raise Exception("Tensorflow version 1.0.0 required")

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        if a.lab_colorization:
            raise Exception("export not supported for lab_colorization")

        input = tf.placeholder(tf.string, shape=[1])
        input_data = tf.decode_base64(input[0])
        input_image = tf.image.decode_png(input_data)
        # remove alpha channel if present
        input_image = input_image[:,:,:3]
        input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
        input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
        batch_input = tf.expand_dims(input_image, axis=0)

        #correct this
        with tf.variable_scope("generator") as scope:
            batch_output = deprocess(create_generator(preprocess(batch_input), 3))

        output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
        output_data = tf.image.encode_png(output_image)
        output = tf.convert_to_tensor([tf.encode_base64(output_data)])

        key = tf.placeholder(tf.string, shape=[1])
        inputs = {
            "key": key.name,
            "input": input.name
        }
        tf.add_to_collection("inputs", json.dumps(inputs))
        outputs = {
            "key":  tf.identity(key).name,
            "output": output.name,
        }
        tf.add_to_collection("outputs", json.dumps(outputs))

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model")
            export_saver.export_meta_graph(filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(a.output_dir, "export"), write_meta_graph=False)

        return

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs are [batch_size, height, width, channels]
    # targets are [batch_size, 12]
    # meta are [batch_size, 12]
    model = create_model(examples.inputs, examples.meta, examples.targets)

    inputs = deprocess(examples.inputs)
    meta   = examples.meta
    targets = examples.targets
    situation = model.situation
    outputs = model.outputs
    next_commands = model.next_commands

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    commands_snes9x = [[['UP','DOWN','LEFT','RIGHT','A','B','X','Y','L','R','START','SELECT']]]
#    commands_snes9x = tf.Print(commands_snes9x,[commands_snes9x],"commands_snes9x:",summarize=1000)
    commands_snes9x1 = tf.tile(commands_snes9x,[a.batch_size,a.frames,1])
#    commands_snes9x1 = tf.Print(commands_snes9x1,[commands_snes9x1],"commands_snes9x(1):",summarize=1000)

    def convert_commands(commands):
        fire_level = tf.fill(commands.get_shape(),a.command_fire_level)
#        fire_level = tf.Print(fire_level,[fire_level],'fire_level:',summarize=100)       
#        commands   = tf.Print(commands,[commands],'commands:',summarize=100)
        conditions = tf.greater_equal(commands, fire_level) 
#        conditions = tf.Print(conditions,[conditions],"conditions:",summarize=100)
        empty      = tf.fill(commands.get_shape(),'')
        converted_commands = tf.where(conditions, commands_snes9x1, empty)
#        converted_commands = tf.Print(converted_commands,[converted_commands],"converted_commands:",summarize=100)
        converted_commands = tf.reduce_join(converted_commands, 2, separator=' ')
#        converted_commands = tf.Print(converted_commands,[converted_commands],"converted_commands:",summarize=100)
        converted_commands = tf.reduce_join(converted_commands, 1, separator='\n')
#        converted_commands = tf.Print(converted_commands,[converted_commands],"converted_commands:",summarize=100)

        return converted_commands

    with tf.name_scope("convert_outputs"):
        converted_next_commands = convert_commands(next_commands)
        converted_next_commands = tf.Print(converted_next_commands,[converted_next_commands],"next_commands:",summarize=100)

    def convert_meta(meta_to_convert):
        if meta_to_convert is not None:
            converted_meta = tf.to_int32(meta_to_convert)
            converted_meta = tf.as_string(converted_meta)
            converted_meta = tf.reduce_join(converted_meta,1,separator='\t')
#            converted_meta = tf.Print(converted_meta,[converted_meta],"converted_meta:",summarize=100)
            return converted_meta
        else:
            return []

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "meta": convert_meta(meta),
            "outputs": convert_meta(outputs),
            "targets": convert_meta(targets),
            "next_commands": converted_next_commands,
            "situation": convert_meta(situation),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("meta_summary"):
        tf.summary.tensor_summary("meta", meta)

    with tf.name_scope("targets_summary"):
        if targets is not None:
            tf.summary.tensor_summary("targets", targets)

    with tf.name_scope("situation_summary"):
        tf.summary.tensor_summary("situation", situation)

    with tf.name_scope("outputs_summary"):
        tf.summary.tensor_summary("outputs", outputs)

    with tf.name_scope("next_commands"):
        tf.summary.tensor_summary("next_commands", next_commands)

    with tf.name_scope("generator_loss"):
        if model.gen_loss is not None:
            tf.summary.scalar("generator_loss", model.gen_loss)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    if model.gen_grads_and_vars is not None:
        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)

            print("wrote index at", index_path)
        else:
            # training
            start = time.time()

            #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(a.progress_freq):
                    fetches["gen_loss"] = model.gen_loss

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss", results["gen_loss"])

                if should(a.save_freq) and not examples.targets is None:
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

main()
