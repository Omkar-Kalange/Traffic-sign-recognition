import numpy as np
from PIL import Image
import pickle
from tensorflow import keras
from tkinter import Tk, filedialog
import warnings
warnings.filterwarnings("ignore")

#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signal', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }



try:
    root = Tk()
    filename = filedialog.askopenfilename( title = "Select a File",  filetypes = (("JPEG", "*.jpeg; *.jpg"), ("All files", "*.*")))
    root.destroy()

    image = Image.open(filename)
    image = image.resize((30,30))
    image = np.array(image)
    #image= image.flatten()
        
    classifier = keras.models.load_model('cnn_model')
        
    predict = classifier.predict(np.array([image]))
    
    class_of_image = np.argmax(predict)
     
    print("\nTraffic sign : ",classes[class_of_image], "\n\n")
       
except:
    print("Error in loading image...")