import cv2
import numpy as np
from PIL import Image


face_classifier= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

cfg_file_path='yolov3_last_training.cfg'
weights_file_path='yolov3_training_last.weights'
names_file_path = 'names.names'

def nudity_blur(img, cfg_file, weight_file, name_file):
	'''returns the censored image, label for the part and confidence for that part'''
	classes=[]
	with open(name_file, 'r') as f:
		classes=[line.strip() for line in f.readlines()]
    # give configuration and weight file
	net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file) 
	height, width, channels = img.shape
    # convert image to blob
	blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0),swapRB=True, crop=False )
    # feeding blob input to yolo input
	net.setInput(blob)
    # getting last layer
	last_layer = net.getUnconnectedOutLayersNames()
    # getting output from this layer
	last_out = net.forward(last_layer)
    
	boxes=[]         # for storing coordinates of rectangle
	confidences=[]   # for storing probabilities
	class_ids=[]     # for storing the label index
    
    
	for output in last_out:
		for detection in output:
			score = detection[4:]                 # probabilities are after 5th element first 4 are cooordinates
			class_id = np.argmax(score)           # gives index of highest probability
			confidence = score[class_id]          # gives the highest probability
			if confidence > 0.05:                  # if the probability of happening is above 20%
				center_x = int(detection[0] * width)
				center_y = int(detection[1] * height)
				w = int(detection[2] * width)
				h = int(detection[3] * height)

				x = int(center_x - w/2)
				y = int(center_y - h/2)

				boxes.append([x,y,w,h])
				confidences.append((float(confidence)))
				class_ids.append(class_id)
                
	labels=[]
	conf=[]
	indices= np.array(cv2.dnn.NMSBoxes(boxes, confidences, 0.05,0.4))
#     indices = np.array(index)
	for i in indices.flatten():
		x,y,w,h = boxes[i]                              # returns coordinates
		label = str(classes[class_ids[i]])              # returns label for each image
		confidence = str(round(confidences[i],2))       # returns confidence for each label
        # make blur
      
        #roi = img[y:y+h, x:x+w]
		img[y:y+h, x:x+w]=cv2.medianBlur(img[y:y+h, x:x+w], ksize=91)
		labels.append(label)
		conf.append(confidence)
    
    
	return(img, labels, conf)

def blur_face(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces=face_classifier.detectMultiScale(gray, 1.3,3)
	if faces == ():
		# return img
		print('No Face Detected')
	else:
		for x,y,w,h in faces:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.3,3)
	if eyes == ():
		# return img
		print('No Face Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.3,3)
	if smile == ():
		# return img
		print('No Face Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img


## Deployiong web app on sreamlit

import streamlit as st

def about():
	st.markdown('''This Web App is made for censoring(blurring) the NSWF(Not Safe For Work) materials, that includes personal pictures depicting Nudity.
		This App will blur the inappropriate areas and body. Along with that we have provided option for blurrring Eyes, Smile, Face and Nudity.''')

	st.markdown('''YOLO custom object dedection is used for detection of Nudity whereas HAAR Cascade Classifier is used for detecting Face, Smile and Eyes.''')


def main():
	st.title("PCOCM: Parental Control Over Children's Media")
	st.write('Using YOLOV3 object detection and Haar Cascade Classifier we detect the NSWF parts and blur them with OpenCV')

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox('Select an option', activities)

	if choice == 'Home':
		st.write('Go to the about section to know more about it')

		image_file=st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'webp'])

		if image_file is not None:
			image = Image.open(image_file)
			image = np.array(image)
            # image.setflags(write=True)

			choice_type = st.sidebar.radio('Make a choice', ('Original','Eyes', 'Face', 'Smile', 'Nudity'))

			if st.button('Process'):
				if choice_type == 'Original':
					result_image = image
					st.image(result_image, use_column_width=True)
					st.info(f'{choice_type} image returned')
					
				elif choice_type == 'Eyes':
					result_image= blur_eyes(image)
					st.image(result_image, use_column_width=True)
					st.info(f'{choice_type} of the image got blurred')
					# st.info(blur_eyes(image))	
				elif choice_type == 'Face':
					result_image= blur_face(image)
					st.image(result_image, use_column_width=True)
					st.info(f'{choice_type} of the image got blurred')
					# st.info(blur_face(image))
				elif choice_type == 'Smile':
					result_image= blur_smile(image)
					st.image(result_image, use_column_width=True)
					st.info(f'{choice_type} of the image got blurred')
					# st.info(blur_smile(image))
				elif choice_type == 'Nudity':
					result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
					st.image(result_image, use_column_width=True)
					if len(confidence) ==0:
						st.info('No nudity present in the image.')
					else:
						st.info(f'Nudity percentage: {np.mean([float(i) for i in confidence])*100}%')
		# choice_type = st.sidebar.radio('Make a choice', ['Eyes', 'Face', 'Smile', 'Nudity'])

	elif choice == 'About':
		about()

if __name__ == '__main__':
	main()