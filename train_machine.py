from keras.models import Sequential, Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import keras, os, sys, argparse, pickle, glob

# In[]:
def identify_input():
    """Parses the inputs provided by the user"""

    p = argparse.ArgumentParser(description = """
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        Some intilligent and up to date help info should go here
        +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """,
        
        epilog = """
        Some additional useful info should go here.
        """)

    p.add_argument('dir'              , type = str)  
    p.add_argument('name'             , type = str)
    p.add_argument('-epochs', '-e'    , type = int, default = 1)
    p.add_argument('-batchsize', '-bs', type = int, default = 12)
    p.add_argument('-machine', '-m'   , type = str, default = 'vgg16', choices = ['vgg16','vgg19','xception','inceptionv3','inceptionresnetv2'])
    
    if len(sys.argv) == 1:
        print('No input provided, exiting...')
        sys.exit()

    S = p.parse_args()

    return S

settings = identify_input()


def get_model(settings, N):
       
    # Load pre-trained model
    if settings.machine == 'vgg16':
        nx, ny = 224, 224
        pretrain = keras.applications.vgg16.VGG16(weights='imagenet',include_top = False)
    elif settings.machine == 'vgg19':
        nx, ny = 224, 224
        pretrain = keras.applications.vgg19.VGG19(include_top = False, input_shape = (331, 331, 3))
    # NASNET doesn't currently work correctly, will be updated in future version of Keras apparently
    #elif settings.machine == 'nasnetlarge':
    #    nx, ny = 331, 331
    #    pretrain = keras.applications.nasnet.NASNetLarge(weights='imagenet',include_top = False)
    elif settings.machine == 'xception':
        nx, ny = 299, 299
        pretrain = keras.applications.xception.Xception(weights='imagenet',include_top = False)
    elif settings.machine == 'inceptionv3':
        nx, ny = 299, 299    
        pretrain = keras.applications.inception_v3.InceptionV3(weights='imagenet',include_top = False)
    elif settings.machine == 'inceptionresnetv2':
        nx, ny = 299, 299
        pretrain = keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',include_top = False)

    for layer in pretrain.layers:
        layer.trainable = False
 
    x = pretrain.output
    
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
   
    predictions = Dense(N, activation='softmax')(x)

    model = Model(inputs = pretrain.input, outputs = predictions)
    
    return model, nx, ny

# In[ ]:
# The training set is used by the machine to optimize on
training_dir    = os.path.join(*[settings.dir,'training'])

# The validation set is not used by the machine, but by the user to optimize hyperparameters
validation_dir  = os.path.join(*[settings.dir,'validation'])

# Path to store fit
machine_path = os.path.join(*[settings.dir,'%s_%s.h5' % (settings.machine, settings.name)])

# In[ ]:
# Initialize training data set class
train_data_gen = ImageDataGenerator(rescale = 1./255)

valid_data_gen = ImageDataGenerator(rescale = 1./255)

nepochs = settings.epochs

batch_size = settings.batchsize

N_classes = len([x for x in glob.glob(os.path.join(*[training_dir,'*/']))])

# Get model architecture (minus final output layer)
model, nx, ny = get_model(settings, N_classes)

# Training set generator is initialized. This generates the batches as they are needed during training
print('Training set: %s' % (training_dir))
training_set = train_data_gen.flow_from_directory(training_dir, 
                                                  target_size = (nx, ny), 
                                                  batch_size = batch_size, 
                                                  class_mode = 'categorical', 
                                                  shuffle = True)

# Validation set generator is initialized. This generates the batches as they are needed during training
print('Validation set: %s' % (validation_dir))
validation_set = valid_data_gen.flow_from_directory(validation_dir, 
                                                    target_size = (nx, ny), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'categorical', 
                                                    shuffle = True)

# In[ ]:
# Compile model with specified loss function
#opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = SGD(lr = 0.001, decay = 1e-4, nesterov = False)
#opt = Adam(lr = 0.0001)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

# In[ ]:Train the model
checkpoint = [ModelCheckpoint(filepath = machine_path)]

history = model.fit_generator(training_set, 
                    steps_per_epoch = int(training_set.samples/batch_size),
                    validation_data = validation_set,
                    validation_steps = int(validation_set.samples/batch_size),
                    shuffle = True,
                    epochs = nepochs,
                    callbacks = checkpoint)

pickle.dump(history, open(machine_path.replace('.h5','.history'), "wb" ) )

# In[ ]: Save model to file 
model.save(machine_path)