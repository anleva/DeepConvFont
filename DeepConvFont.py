from keras.layers import Input, Dense, MaxPooling3D, UpSampling3D, Conv3D, Flatten, Layer
from keras.layers import Lambda, Reshape, Permute, LocallyConnected2D, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.objectives import binary_crossentropy
from keras import backend as K

import h5py
import numpy as np
import datetime as dt
import scipy.misc
from os import listdir

# Global references
DATA_FILE_NAME = r'.\data\fonts.hdf5'
OUT_FOLDER = r'.\data\\'
MODEL_FILE_NAME = 'dcf_v3'


# Net Architecture
CHARS_IN_FONT = 26
CHAR_CONV_DIMS = [1, 16, 16, 16, 16, 32, 32]
FONT_REDUCE_DIMS = [26, 128, 32]
DENSE_DIMS = [1024, 524, 256, 128, 64]


# Other net Parameters
TRAIN_MODEL = 'CHAR_NO_SAMPLE'  # 'FONT' or 'CHAR' or 'CHAR_NO_SAMPLE'
VAE_BETA_CHAR = 0
VAE_BETA_FONT = 0
NUMBER_OF_FONTS = 100
BATCH_SIZE = 1
LOAD_PARAMS = True
EPOCHS_BETWEEN_SAVES = 10
NUMBER_OF_SAVES = 50
VERBOSE = True


# Intermediate calculations
CHAR_ENCODING_DIM = CHAR_CONV_DIMS[-1]
FONT_ENCODING_DIM = DENSE_DIMS[-1]
FONT_DIM_PRE_CONDENSE = CHAR_CONV_DIMS[-1] * FONT_REDUCE_DIMS[-1]
CHAR_REDUCE_DIM = int(64 / (2 ** (len(CHAR_CONV_DIMS)-1)))


# Define variational sampling
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = list(K.int_shape(z_mean))
    dim[0] = batch
    rand_norm = K.random_normal(shape=tuple(dim))

    return z_mean + K.exp(0.5 * z_log_var) * rand_norm


# Define a layer that adds KL divergence (multiplied by VAE_BETA) to the final loss
class KLDivergenceLayer(Layer):
    """ 
    Identity transform layer that adds KL divergence to the final model loss.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        self.vae_beta = kwargs.get('vae_beta', 1.0)
        del kwargs['vae_beta']
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = -0.5 * (1 + log_var - K.square(mu) - K.exp(log_var))

        # Since we have multiple encoding levels, with very different dimensions, summing all dimensions except batch
        # will lead to very different impact on loss. The difference between the optimal value of VAE_BETA_FONT and
        # VAE_BETA_CHAR would likely be big.
        # Therefore, perhaps we should just take the mean over all dims, instead of summing some.

        while len(kl_batch.shape) >= 2:
            kl_batch = K.sum(kl_batch, axis=-1)  # Sum over all axes, except the batch axis

        self.add_loss(self.vae_beta * K.mean(kl_batch), inputs=inputs)
        return inputs


#############################
#                           #
#   Define CHAR  Encoder    #
#                           #
#############################

# Input data
# Dimensions are (batch * nb_characters * pixel_width * pixel_height)
# Shape is       (56443 *       62      *      64     *      64     )
char_encoder_input = Input(shape=(CHARS_IN_FONT, 64, 64, 1), name='Char_Encoder_Input')  # 56443 * 62 * 64 * 64
x = char_encoder_input

# Convolutional part
i = 1
for f in CHAR_CONV_DIMS[1:-1]:
    x = Conv3D(f, (1, 3, 3), activation='relu', padding='same', name='E_C_Cov_' + str(i))(x)
    x = MaxPooling3D((1, 2, 2), padding='same', name='E_C_Pool_' + str(i))(x)
    i += 1

# Beta-VAE part
z_c_mean = Conv3D(CHAR_ENCODING_DIM, (1, 3, 3), padding='same', kernel_regularizer=l2(0.01), name='E_C_Enc_Mean_Conv')(x)
z_c_mean = MaxPooling3D((1, 2, 2), padding='same', name='E_C_Enc_Mean_Pool')(z_c_mean)

z_c_log_var = Conv3D(CHAR_ENCODING_DIM, (1, 3, 3), padding='same', kernel_regularizer=l2(0.01), name='E_C_Enc_Var_Conv')(x)
z_c_log_var = MaxPooling3D((1, 2, 2), padding='same', name='E_C_Enc_Var_Pool')(z_c_log_var)

z_c_sample = Lambda(sampling, output_shape=(CHARS_IN_FONT, 1, 1, CHAR_ENCODING_DIM,),
                    name='E_C_Enc_Sample')([z_c_mean, z_c_log_var])

kl_params = {'vae_beta': VAE_BETA_CHAR}
z_c_mean, z_c_log_var = KLDivergenceLayer(name='Char_VAE', **kl_params)([z_c_mean, z_c_log_var])

# Define char encoder
char_encoder = Model(char_encoder_input, [z_c_mean, z_c_log_var, z_c_sample], name='Char_Encoder')
if VERBOSE:
    char_encoder.summary()


#############################
#                           #
#   Define FONT  Encoder    #
#                           #
#############################

font_encoder_input = Input(shape=(CHARS_IN_FONT, 1, 1, CHAR_ENCODING_DIM,), name='Font_Encoder_Input')
x = font_encoder_input

# Restructure, to 2d+feature, with chars as features: (CHAR_ENCODING_DIM, 1, CHARS_IN_FONT)
x = Reshape((CHARS_IN_FONT, CHAR_ENCODING_DIM, 1), name='E_F_Reshape')(x)
x = Permute((2, 3, 1), name='E_F_Permute')(x)

# Locally connected Conv part, equivalent to one Deep Dense net per encoding dimension
i = 1
for f in FONT_REDUCE_DIMS[1:]:
    x = LocallyConnected2D(f, (1, 1), name='E_F_Conv_' + str(i))(x)
    i += 1

# Condense with fully connected layers
x = Flatten(name='E_F_Flatten')(x)
i = 1
for d in DENSE_DIMS[1:-1]:
    x = BatchNormalization(name='E_F_BatchNorm_' + str(i))(x)
    x = Dense(d, kernel_regularizer=l2(0.01), name='E_F_Dense_' + str(i))(x)
    i += 1

# Variational encoding on font Level
z_f_mean = Dense(FONT_ENCODING_DIM, kernel_regularizer=l2(0.01), name='E_F_Dense_Enc_Mean')(x)
z_f_log_var = Dense(FONT_ENCODING_DIM, kernel_regularizer=l2(0.01), name='E_Dense_Enc_Var')(x)
z_f_sample = Lambda(sampling, output_shape=(FONT_ENCODING_DIM,), name='E_Enc_Sample')([z_f_mean, z_f_log_var])
kl_params = {'vae_beta': VAE_BETA_FONT}
z_f_mean, z_f_log_var = KLDivergenceLayer(name='Font_VAE', **kl_params)([z_f_mean, z_f_log_var])

# Define font encoder
font_encoder = Model(font_encoder_input, [z_f_mean, z_f_log_var, z_f_sample], name='Font_Encoder')
if VERBOSE:
    font_encoder.summary()


#############################
#                           #
#    Define Font Decoder    #
#                           #
#############################

# Define input
font_decoder_input = Input(shape=(FONT_ENCODING_DIM,), name='Font_Decoder_Input')
y = font_decoder_input

# Upsample with dense layers
i = 1
for d in reversed(DENSE_DIMS[:-1]):
    y = Dense(d, kernel_regularizer=l2(0.01), name='D_F_Dense_' + str(i))(y)
    y = BatchNormalization(name='D_F_BatchNorm_' + str(i))(y)
    i += 1

# Get character-level features
y = Reshape((CHAR_CONV_DIMS[-1], 1, FONT_REDUCE_DIMS[-1]), name='D_F_Reshape_1')(y)
i = 1
for d in reversed(FONT_REDUCE_DIMS[1:-1]):
    y = LocallyConnected2D(d, (1, 1), activation='relu', name='D_F_Conv_' + str(i))(y)
    i += 1

# Get character-level features - Last should be without non-linearity
y = LocallyConnected2D(FONT_REDUCE_DIMS[0], (1, 1), name='D_F_Conv_' + str(i))(y)

# Permute to get dimensions: (CHARACTER, 1, FEATURES)
y = Permute((3, 2, 1), name='D_F_Permute')(y)

# Reshape to (CHARACTER, 1, 1, FEATURES)
char_encoding = Reshape((CHARS_IN_FONT, 1, 1, CHAR_ENCODING_DIM), name='D_F_Reshape_2')(y)

# Define font decoder
font_decoder = Model(font_decoder_input, char_encoding, name='Font_Decoder')
if VERBOSE:
    font_decoder.summary()


#############################
#                           #
#    Define Char Decoder    #
#                           #
#############################

# Define input
char_decoder_input = Input(shape=(CHARS_IN_FONT, 1, 1, CHAR_ENCODING_DIM,), name='Char_Decoder_Input')
y = char_decoder_input

# Upsample with Conv3D
i = 1
for d in reversed(CHAR_CONV_DIMS[1:-1]):
    y = UpSampling3D(size=(1, 2, 2), name='D_C_Upsample_' + str(i))(y)
    y = Conv3D(d, (1, 3, 3), activation='relu', padding='same', name='D_C_Cov_' + str(i))(y)
    i += 1

# Use sigmoid activation for last layer
y = UpSampling3D(size=(1, 2, 2), name='D_C_Upsample_' + str(i))(y)
y = Conv3D(CHAR_CONV_DIMS[0], (1, 3, 3), activation='sigmoid', padding='same', name='D_C_Cov_' + str(i))(y)

# Define decoder
chars_in_font = y
char_decoder = Model(char_decoder_input, chars_in_font, name='Char_Decoder')
if VERBOSE:
    char_decoder.summary()


#############################
#                           #
#  Define Font Auto-Encoder #
#                           #
#############################

ae_font = \
    char_decoder(
        font_decoder(
            font_encoder(
                char_encoder(
                    char_encoder_input
                )[-1]
            )[-1]
        )
    )

font_autoencoder = Model(char_encoder_input, ae_font, name='font_ae')
font_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
if VERBOSE:
    font_autoencoder.summary()


#############################
#                           #
#  Define Char Auto-Encoder #
#                           #
#############################

ae_char = \
    char_decoder(
        char_encoder(
            char_encoder_input
        )[-1]
    )


char_autoencoder = Model(char_encoder_input, ae_char, name='char_ae')
char_autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
if VERBOSE:
    char_autoencoder.summary()


#############################
#                           #
#  Define Char Auto-Encoder #
#      without sampler      #
#                           #
#############################

ae_char = \
    char_decoder(
        char_encoder(
            char_encoder_input
        )[0]
    )

char_autoencoder_no_sample = Model(char_encoder_input, ae_char, name='char_ae_no_sample')
char_autoencoder_no_sample.compile(optimizer='adam', loss='binary_crossentropy')
if VERBOSE:
    char_autoencoder_no_sample.summary()



#############################
#                           #
#     Data & Utilities      #
#                           #
#############################

fonts = h5py.File(DATA_FILE_NAME, 'r')
fonts_group = fonts['fonts']


def get_np_dataset(f0, f1):
    return fonts_group[f0:f1, 0:CHARS_IN_FONT, :, :].reshape((f1-f0, CHARS_IN_FONT, 64, 64, 1)) / 255.0


def save_progress(model):
    # Save model
    time_stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_fn = OUT_FOLDER + MODEL_FILE_NAME + '_{0}_{1}.h5'.format(model.name, time_stamp)
    model.save(model_fn)
    print('Model saved.')

    # Save images with example fonts
    font_out = model.predict(get_np_dataset(0, 5))
    for font in range(font_out.shape[0]):
        img = list()
        for char in range(font_out.shape[1]):
            img_char = font_out[font, char, :, :, 0] * 255.0
            img.append(img_char)
        img = np.hstack(img)
        font_file_name = r'{0}\img\font_{1}_{2}_timestamp_{3}.jpg'.format(OUT_FOLDER, font, model.name, time_stamp)
        scipy.misc.imsave(font_file_name, img)
    print('Images saved.')


current_dataset = get_np_dataset(0, NUMBER_OF_FONTS)
m_pixels = np.round(np.prod(current_dataset.shape) / (6e4 * 28**2), decimals=2)
m_characters = np.round(current_dataset.shape[0] * current_dataset.shape[1] / 6e4, decimals=2)
print('Current dataset is {0} times the size of MNIST, in terms of pixels.\n'
      'Current dataset is {1} times the size of MNIST, in terms of characters.'.format(
        m_pixels, m_characters))


# Select Model
if TRAIN_MODEL == 'FONT':
    model_to_train = font_autoencoder
elif TRAIN_MODEL == 'CHAR':
    model_to_train = char_autoencoder
elif TRAIN_MODEL == 'CHAR_NO_SAMPLE':
    model_to_train = char_autoencoder_no_sample


if LOAD_PARAMS:
    model_fn = MODEL_FILE_NAME + '_' + model_to_train.name
    model_files = [f for f in listdir(OUT_FOLDER) if f[:len(model_fn)] == model_fn]
    if len(model_files) > 0:
        model_files.sort()
        model_to_train.load_weights(OUT_FOLDER + model_files[-1])
        print('Model {} loaded.'.format(model_files[-1]))
    else:
        print('No model found. Starting training from scratch.')


for i in range(NUMBER_OF_SAVES):
    model_to_train.fit(current_dataset, current_dataset,
                       epochs=EPOCHS_BETWEEN_SAVES,
                       batch_size=BATCH_SIZE,
                       shuffle=True)

    save_progress(model_to_train)


print('Terminated at end.')

