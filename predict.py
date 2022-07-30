# Imports
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from os import listdir
from os.path import isfile, join
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Methods
def prepareImg(image, label, imgShape):
    image = tf.cast(image, tf.float32) # Typecast raw image values to float32
    image = tf.image.resize(image, tuple(imgShape)) # Overwrite image with resized image
    image /= 255.0 # Normalize to maximum value of byte which is used to display R/G/B channels ((2^8)-1 = 255)
    return image, label

def process_image(npImg, imgShape):
  tensor = tf.convert_to_tensor(npImg) # Convert NP array to TF tensor
  prepImg, dummy = prepareImg(tensor, None, imgShape) # Returns resized + normalized image
  return prepImg.numpy()

def predict(image_path, model, top_k):
  # Load image from path and do pre-processing
  loadedImg = Image.open(image_path)
  loadedNpImg = np.asarray(loadedImg)
  inputLayerShape = model.layers[0].input_shape[1:3] # Get shape of input layer of Kera model, example for .input_shape[]: (None, 224, 224, 3)
  procImg = np.expand_dims(process_image(loadedNpImg, inputLayerShape), axis=0) # Add extra dimension at position 0 (=first position, axis does not mean row/col here)

  # Perform feedforward prediction to retrieve probs
  predProbs = model.predict(procImg) # get probabilities for classes given the provided image
  
  # Provide probabilities and classes
  topKsIdx = (-1 * predProbs[0]).argsort()[:top_k] # Get indices of top k (highest) probabilities; use negative numbers for all vals to get correct order
  probs = [predProbs[0][i] for i in topKsIdx] # contains top k probs
  classes = topKsIdx + 1 # add +1 to represent class number as there is no class '0'
  return probs, classes

# Parse arguments from CLI
parser = argparse.ArgumentParser(description='Keras Predictor - script that can predict probabilities and classes for given images')

parser.add_argument(dest='imgPath', action='store', type=str, help='File path to the image to be classified')
parser.add_argument(dest='modelPath', action='store', type=str, help='File path to the image to be classified')
parser.add_argument('--top_k', dest='top_k', type=int, action='store', default=1,\
                    help='Return the top k most likely classes')
parser.add_argument('--category_names', dest='jsonPath', type=str, action='store', default='',\
                    help='Add file path to JSON file to map class IDs to category names')
    
args = vars(parser.parse_args()) # Store args

# Get all absolute files in image directory
imgPath = args['imgPath']
absFilePaths = [join(imgPath,f) for f in listdir(imgPath) if isfile(join(imgPath,f))] # Check for all folders/files in dir and only use files

# Load model
try:
    model = tf.keras.models.load_model(args['modelPath'], custom_objects={'KerasLayer':hub.KerasLayer})
except:
    print("Not able to load the Keras model from the given path. Please check and try again.")
    sys.exit() # Exit due to unsolveable isse

# Predict for all files in directory
for imgFilePath in absFilePaths:
    try:
        probs, classes = predict(imgFilePath,model,args['top_k'])
    except:
        print("Not able to make predictions for the given images path. Please check and try again.")
        sys.exit() # Exit due to unsolveable isse
    
    # Get class names from json if argument was passed     
    class_names = {}
    try:
        if args['jsonPath'] != '':
            with open(args['jsonPath'], 'r') as f:
                class_names = json.load(f)
    except:
        print("Not able to load the category names from the given JSON path. Please check and try again.")
        sys.exit() # Exit due to unsolveable isse
        
    # Print corresponding file path, predictions and classes / class names
    print("Evaluated sample: {}".format(imgFilePath))
    for i in range(len(probs)):
        if len(class_names) > 0: # JSON class names defined
            print("Rank #{}: Class '{}' with a probability of {:.0%}".format(i,class_names[str(classes[i])],probs[i]))
        else:
            print("Rank #{}: Class '{}' with a probability of {:.0%}".format(i,classes[i],probs[i]))