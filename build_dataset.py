from Helper import config
from imutils import paths
import random
import shutil
import os

imagePaths = sorted(list(paths.list_images(config.ORIG_INPUT_DATASET)))
random.seed(42)
random.shuffle(imagePaths)

# compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# we'll be using part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
validPaths = trainPaths[:i]
trainPaths =trainPaths[i:]

datasets = [("training" ,trainPaths ,config.TRAIN_PATH) ,("testing" ,testPaths ,config.TEST_PATH) ,
                ("validation" ,validPaths ,config.VAL_PATH)]

for (dType ,imagePaths ,baseOutput) in datasets:
    print("[INFO] buiding {} split".format(dType))

    if not os.path.exists(baseOutput):
        print("[INFO] creating {} directory".format(baseOutput))
        os.makedirs(baseOutput)

    for imagePath in imagePaths:
        filename = imagePath.split(os.path.sep)[-1]
        label = imagePath.split(os.path.sep)[-2]

        labelPath = os.path.sep.join([baseOutput ,label])

        if not os.path.exists(labelPath):
            print("[INFO] creating {} directory".format(labelPath))
            os.makedirs(labelPath)

        # construct the path to the destination image and then copy
		# the image itself
        p = os.path.sep.join([labelPath ,filename])
        shutil.copy2(imagePath ,p)
