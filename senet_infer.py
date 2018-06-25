from model import *
from data_import import *

import sys, getopt

valfolder = "dataset/valset_noisy"
modfolder = "models"

try:
    opts, args = getopt.getopt(sys.argv[1:],"hd:m:",["ifolder=,modelfolder="])
except getopt.GetoptError:
    print 'Usage: python senet_infer.py -d <inputfolder> -m <modelfolder>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'Usage: pythonsenet_infer.py -d <inputfolder> -m <modelfolder>'
        sys.exit()
    elif opt in ("-d", "--inputfolder"):
        valfolder = arg
    elif opt in ("-m", "--modelfolder"):
        modfolder = arg
print 'Input folder is "' + valfolder + '/"'
print 'Model folder is "' + modfolder + '/"'

if valfolder[-1] == '/':
    valfolder = valfolder[:-1]

if not os.path.exists(valfolder+'_denoised'):
    os.makedirs(valfolder+'_denoised')

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 6 # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

fs = 16000

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1])
        
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

# LOAD DATA
valset = load_noisy_data_list(valfolder = valfolder)
valset = load_noisy_data(valset)

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print "Config ready"

sess.run(tf.global_variables_initializer())

print "Session initialized"

saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
saver.restore(sess, "./%s/se_model.ckpt" % modfolder)

#####################################################################################

for id in tqdm(range(0, len(valset["innames"]))):

    i = id # NON-RANDOMIZED ITERATION INDEX
    inputData = valset["inaudio"][i] # LOAD DEGRADED INPUT

    # VALIDATION ITERATION
    output = sess.run([enhanced],
                        feed_dict={input: inputData})
    output = np.reshape(output, -1)
    wavfile.write("%s_denoised/%s" % (valfolder,valset["shortnames"][i]), fs, output)

