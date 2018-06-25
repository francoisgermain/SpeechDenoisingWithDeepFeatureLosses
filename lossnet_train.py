from model import *
from data_import import *

import sys, getopt

# COMMAND LINE OPTIONS
outfolder = "."
try:
    opts, args = getopt.getopt(sys.argv[1:],"ho:",["outfolder="])
except getopt.GetoptError:
    print 'Usage: python senet_infer.py -o <outfolder>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'Usage: python senet_infer.py -o <outfolder>'
        sys.exit()
    elif opt in ("-o", "--outfolder"):
        outfolder = arg
print 'Output model folder is "' + outfolder + '/"'

# FEATURE LOSS NETWORK PARAMETERS
LOSS_LAYERS = 14 # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = 5 # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM = "SBN" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

#################################################################################################################
# NETWORK SETUP #################################################################################################

n_tasks = 0 # NUMBER OF TASKS
n_classes = [] # NUMBER OF CLASSES PER TASK
error_type = [] # TYPE OF ERROR PER TASK
label_task = [] # GROUND TRUTH LABELS PLACEHOLDER
conv_task = [] # LINEAR LAYER PER TASK
pred_task = [] # LOGISTIC PREDICTION OUTPUT LAYER PER TASK
loss_task = [] # LOSS OUTPUT LAYER PER TASK
opt_task = [] # NETWORK OPTIMIZER PER TASK

# DATA LOADING ####################################################################################################

names = [] # FILE NAMES PER TASK
labels = [] # FILE LABELS PER TASK
datasets =[] # AUDIO FILES PER TASK
label_lists = [] # LABEL LISTS PER TASK
sets = ['train','val']

# ACOUSTIC SCENE CLASSIFICATION TASK
n_tasks += 1
asc_datasets, asc_labels, asc_names, asc_label_list =\
    load_asc_data("dataset/asc/")
n_classes.append(len(asc_label_list))
error_type.append(1) # CLASSIFICATION ERROR
names.append({})
labels.append({})
datasets.append({})
for setname in sets:
    names[n_tasks-1][setname] = asc_names[setname]
    labels[n_tasks-1][setname] = asc_labels[setname]
    datasets[n_tasks-1][setname] = asc_datasets[setname]
label_lists.append(asc_label_list)

# DOMESTIC AUDIO TAGGING TASK
n_tasks += 1
dat_datasets, dat_labels, dat_names, dat_label_list =\
    load_dat_data("dataset/dat/")
n_classes.append(len(dat_label_list))
error_type.append(2) # TAGGING ERROR
names.append({})
labels.append({})
datasets.append({})
for setname in sets:
    names[n_tasks-1][setname] = dat_names[setname]
    labels[n_tasks-1][setname] = dat_labels[setname]
    datasets[n_tasks-1][setname] = dat_datasets[setname]
label_lists.append(dat_label_list)

##################################################################################################################
# INITIALIZE CLASSIFICATION NETWORKS

# INPUT SIGNAL PLACEHOLDER
input=tf.placeholder(tf.float32,shape=[1,1,None,1])

# DEEP FEATURE NETWORK
features = lossnet(input, n_layers=LOSS_LAYERS, norm_type=LOSS_NORM,
               base_channels=LOSS_BASE_CHANNELS, blk_channels=LOSS_BLK_CHANNELS)

# TASK SPECIFIC LAYERS
for id in range(n_tasks):

    # OUTPUT LABEL LAYER
    label_task.append(tf.placeholder(tf.float32,shape=[1,n_classes[id]]))
    # AVERAGE POOLING OF FEATURES
    avg_features = tf.reduce_mean(features[-1], axis=2, keep_dims=True)
    # LINEAR LAYER
    net = slim.conv2d(avg_features, n_classes[id], [1,1], activation_fn=None, scope='pred_conv_%d'%id)
    conv_task.append(net)

    # CLASSIFIER OUTPUT LAYER LOGITS
    logits = tf.reshape(conv_task[id], [tf.shape(input)[0],n_classes[id]])
    
    if error_type[id] == 1:
        print "Task %d: Softmax" % id
        pred_task.append(tf.nn.softmax(logits))
        # LOSS LAYER WITH LOGISTIC NON-LINEARITY
        net = tf.nn.softmax_cross_entropy_with_logits(labels=label_task[id], logits=logits)
        # AVERAGE LOSS ACROSS CLASSES
        loss_task.append(tf.reduce_mean(net))
    else:
        print "Task %d: Sigmoid" % id
        pred_task.append(tf.nn.sigmoid(logits))
        # LOSS LAYER WITH LOGISTIC NON-LINEARITY
        net = tf.nn.sigmoid_cross_entropy_with_logits(labels=label_task[id], logits=logits)
        # AVERAGE LOSS ACROSS CLASSES
        loss_task.append(tf.reduce_mean(net))
    
    opt_task.append(tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss_task[id],var_list=[var for var in tf.trainable_variables()]))


#################################################################################################################
# BEGIN SCRIPT #########################################################################################################

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

Nepoch = 2500
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())

#################################################################################################################
# EPOCH INITIALIZATION

train_all = [] # ERROR PER TRAINING FILE
test_all = [] # ERROR PER TESTING FILE
thres_all = [] # ERROR THRESHOLD PER LABEL FOR EQUAL ERROR RATE

for ntask in range(n_tasks):
    train_all.append(np.zeros(len(names[ntask]['train'])))
    test_all.append(np.zeros(len(names[ntask]['val'])))
    thres_all.append(.5 * np.ones(len(label_lists[ntask])))

MAX_NUM_FILE = 0 # MAXIMUM NUMBER OF FILES PER TASK
train_pred_all = [] # UPDATE TRAINING PREDICTED LABEL
train_label_all = [] # UPDATE TRAINING TRUE LABEL
test_pred_all = [] # UPDATE TESTING PREDICTED LABEL
test_label_all = [] # UPDATE TESTING TRUE LABEL

# PRE-LOAD TASKS
for ntask in range(n_tasks):
    # UPDATE MAXIMUM NUMBER OF FILES PER TASK
    MAX_NUM_FILE = np.maximum(MAX_NUM_FILE, len(names[ntask]['train']))

    # TRAINING DATA
    train_pred_all.append([]) # UPDATE TRAINING PREDICTED LABEL
    train_label_all.append([]) # UPDATE TRAINING TRUE LABEL
    for nfile in range(len(names[ntask]['train'])):
        train_pred_all[ntask].append(np.zeros((0,len(label_lists[ntask]))))
        train_label_all[ntask].append(np.zeros((0,len(label_lists[ntask]))))

    # TESTING DATA
    test_pred_all.append([]) # UPDATE TESTING PREDICTED LABEL
    test_label_all.append([]) # UPDATE TESTING TRUE LABEL
    for nfile in range(len(names[ntask]['val'])):
        test_pred_all[ntask].append(np.zeros((0,len(label_lists[ntask]))))
        test_label_all[ntask].append(np.zeros((0,len(label_lists[ntask]))))

#################################################################################################################
# EPOCH LOOP

for epoch in range(1,Nepoch+1):

    #################
    # TRAINING LOOP #
    #################
    ids = []
    for ntask in range(n_tasks):
        ids.append(np.random.permutation(len(names[ntask]['train'])))

    # LOOP THRU FILES AND TASKS
    for id in tqdm(range(MAX_NUM_FILE*n_tasks), file=sys.stdout):
    
        ntask = id % n_tasks # TASK NUMBER

        # FILE ID IN THE CURRENT TASK
        file_id = ids[ntask][id % len(ids[ntask])]

        # CORRESPONDING DATA IN CURRENT TASK
        inputData = datasets[ntask]['train'][file_id]
        shape = np.shape(inputData)

        # CULLING PARAMETERS
        width_min = 2**(LOSS_LAYERS+1)-1 # MINIMUM CULLED DURATION
        width_top = 1.*np.size(inputData) # FILE DURATION
        width_max = width_top # MAXIMUM CULLED DURATION

        # RANDOMIZED CULLED LENGTH (LOGARITHMIC WEIGHTING)
        exponent = np.random.uniform(np.log10(width_min-.5), np.log10(width_max+.5))
        width = int(np.round(10. ** exponent))

        # RANDOMIZED STARTING POINT
        n_start = np.random.randint(0, np.size(inputData)-width + 1)

        # CULLED CLIP
        inputData = inputData[:,:,n_start:n_start+width,:]

        # CLIP LABEL
        labelData = np.reshape(1. * np.array([(l in labels[ntask]['train'][file_id]) for l in label_lists[ntask]]), (1, -1))

        # GRADIENT DESCENT UPDATE
        _, pred, current = sess.run([opt_task[ntask], pred_task[ntask], loss_task[ntask]],\
                                    feed_dict={input: inputData, label_task[ntask]: labelData})

        train_all[ntask][file_id] = current
        train_pred_all[ntask][file_id] = np.reshape(pred, [1,-1])
        train_label_all[ntask][file_id] = labelData

    str = "T: %d " % (epoch)
    
    # COMPUTE ERRORS
    for ntask in range(n_tasks):
        str += "%.6f " % (np.mean(train_all[ntask][np.where(train_all[ntask])]))
        if error_type[ntask] == 1: # MEAN CLASSIFICATION ERROR
            str += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(train_pred_all[ntask]),axis=1) == np.argmax(np.vstack(train_label_all[ntask]),axis=1))))
        elif error_type[ntask] == 2: # MEAN EQUAL ERROR RATE
            eer = 0.
            for nl, label in enumerate(label_lists[ntask]):
                thres = np.array([0.,1.,.0])
                fp = 1.
                fn = 0.
                while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
                    thres[-1] = np.mean(thres[:-1])
                    fp = (np.sum((np.vstack(train_pred_all[ntask])[:,nl]>thres[-1])*(1.-np.vstack(train_label_all[ntask])[:,nl])))\
                         / (1e-15+np.sum(1.-np.vstack(train_label_all[ntask])[:,nl]))
                    fn = (np.sum((np.vstack(train_pred_all[ntask])[:,nl]<=thres[-1])*np.vstack(train_label_all[ntask])[:,nl]))\
                         / (1e-15+np.sum(np.vstack(train_label_all[ntask])[:,nl]))
                    if fp < fn:
                        thres[1] = thres[-1]
                    else:
                        thres[0] = thres[-1]
                thres_all[ntask][nl] = thres[-1]
                eer += (fp+fn)/2.
            eer /= len(label_lists[ntask])
            str += "%.6f " % (eer)
    
    print str
    
    if epoch % 25 > 0: # VALIDATION LOOP EVERY 25 TRAINING EPOCHS
        continue

    saver.save(sess, outfolder + "/loss_model.ckpt")

    ###################
    # VALIDATION LOOP #
    ###################
    for ntask in range(n_tasks):
        
        # LOOP THRU FILES
        for id in tqdm(range(len(names[ntask]['val'])), file=sys.stdout):

            # FILE ID AND DATA FOR GIVEN TASK
            file_id = id
            inputData = datasets[ntask]['val'][file_id]
            labelData = np.reshape(1. * np.array([(l in labels[ntask]['val'][file_id]) for l in label_lists[ntask]]), (1, -1))

            # PREDICTION COMPUTATION
            pred, current = sess.run([pred_task[ntask], loss_task[ntask]], feed_dict={input: inputData, label_task[ntask]: labelData})
            
            test_all[ntask][file_id] = current
            test_pred_all[ntask][file_id] = pred
            test_label_all[ntask][file_id] = labelData
            
    str = "V: %d " % (epoch)
    
    # COMPUTE ERRORS
    for ntask in range(n_tasks):
        str += "%.6f " % (np.mean(test_all[ntask][np.where(test_all[ntask])]))
        if error_type[ntask] == 1: # CLASSIFICATION ERROR
            str += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(test_pred_all[ntask]),axis=1) == np.argmax(np.vstack(test_label_all[ntask]),axis=1))))
        elif error_type[ntask] == 2: # EQUAL ERROR RATE
            eer = 0.
            for nl, label in enumerate(label_lists[ntask]):
                thres = np.array([0.,1.,.0])
                fp = 1.
                fn = 0.
                while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
                    thres[-1] = np.mean(thres[:-1])
                    fp = (np.sum((np.vstack(test_pred_all[ntask])[:,nl]>thres[-1])*(1.-np.vstack(test_label_all[ntask])[:,nl]))) / (1e-15+np.sum(1.-np.vstack(test_label_all[ntask])[:,nl]))
                    fn = (np.sum((np.vstack(test_pred_all[ntask])[:,nl]<=thres[-1])*np.vstack(test_label_all[ntask])[:,nl])) / (1e-15+np.sum(np.vstack(test_label_all[ntask])[:,nl]))
                    if fp < fn:
                        thres[1] = thres[-1]
                    else:
                        thres[0] = thres[-1]
                eer += (fp+fn)/2.
            eer /= len(label_lists[ntask])
            str += "%.6f " % (eer)
    
    print str


