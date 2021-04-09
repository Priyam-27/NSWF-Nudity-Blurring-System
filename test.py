import os
import cv2
import cvlib
import numpy as np
from PIL import Image
import streamlit as st
import tempfile
from io import BytesIO
import base64
import requests

eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

id = '1Jz6bmv5AfOq-A0DD_SxevcxSq7VRSEtG'
destination = 'yolo.weights'


download_file_from_google_drive(id, destination)

cfg_file_path='yolov3_last_training.cfg'
weights_file_path=destination
names_file_path = 'names.names'

def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = BytesIO()
	img.save(buffered, format="PNG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/csv;base64,{img_str}" download="image.png">Download result</a>'
	return href
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
	for i in indices.flatten():
		x,y,w,h = boxes[i]                              # returns coordinates
		label = str(classes[class_ids[i]])              # returns label for each image
		confidence = str(round(confidences[i],2))       # returns confidence for each label

		img[y:y+h, x:x+w]=cv2.medianBlur(img[y:y+h, x:x+w], ksize=201)
		labels.append(label)
		conf.append(confidence)
    
    
	return(img, labels, conf)



def face_blur(img):
	coor, _ = cvlib.detect_face(img)
	for face in coor:
		x,y,w,h = face
		roi = img[y:h, x:w]
		img[y:h, x:w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.3,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_eyes_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	eyes=eye_classifier.detectMultiScale(gray, 1.6,5)
	if eyes == ():
		# return img
		print('No Eyes Detected')
	else:
		for x,y,w,h in eyes:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.6,8)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img

def blur_smile_video(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	smile=smile_classifier.detectMultiScale(gray, 1.5,10)
	if smile == ():
		# return img
		print('No Smile Detected')
	else:
		for x,y,w,h in smile:
			roi = img[y:y+h, x:x+w]
			img[y:y+h, x:x+w] = cv2.medianBlur(roi, ksize=151)
	return img


def about():
	st.markdown('''This Web App is made for censoring(blurring) the NSWF(Not Safe For Work) materials, that includes personal pictures depicting Nudity.
		This App will blur the inappropriate areas and body. Along with that we have provided option for blurrring Eyes, Smile, Face and Nudity.''')

	st.markdown('''YOLO custom object dedection is used for detection of Nudity whereas HAAR Cascade Classifier is used for detecting Face, Smile and Eyes.''')


def main():
	st.title("Object Detection and Masking")
	st.subheader('For Recorded as well as Real-time media')
	st.write('Using YOLOV3 object detection and Haar Cascade Classifier we detect the NSWF parts and blur them with OpenCV')

	activities = ['Home', 'About']
	choice = st.sidebar.selectbox('Select an option', activities)

	if choice == 'Home':
		st.write('Go to the about section to know more about it')

		file_type = ['Image', 'Video']
		file_choice = st.sidebar.radio('Select file type', file_type)

		if file_choice == 'Video':
			file = st.file_uploader('Choose file', ['mp4'])

			if file is not None:

				tfile = tempfile.NamedTemporaryFile(delete=False)
				tfile.write(file.read())


				st.sidebar.write('Select the required options')
				option_O = st.sidebar.checkbox('Original')
				option_E = st.sidebar.checkbox('Eyes')
				option_F = st.sidebar.checkbox('Face')
				option_S = st.sidebar.checkbox('Smile')
				option_N = st.sidebar.checkbox('Nudity')

				if st.button('Process'):
					if option_O and any([option_E, option_F, option_N, option_S]):
						st.warning('Cannot show Original and Masked image simultaneously')

					elif option_N and option_E and option_F and option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(face_blur(frame))

					elif option_N and option_E and option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(face_blur(image))

					elif option_N and option_S and option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(face_blur(image))

					elif option_N and option_E and option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(blur_smile_video(blur_eyes_video(image)))

					elif option_F and option_E and option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(face_blur(image))




					elif option_E and option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(face_blur(frame))

					elif option_S and option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(face_blur(frame))

					elif option_N and option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(face_blur(image))

					elif option_N and option_E:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(blur_eyes_video(image))

					elif option_N and option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(blur_smile_video(image))	

					elif option_E and option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							
							stframe.image(blur_eyes_video(blur_smile_video(image)))



					elif option_O:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(frame)

					elif option_E:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(blur_eyes_video(frame))


					elif option_F:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(face_blur(frame))


					elif option_S:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							stframe.image(blur_smile_video(frame))



					elif option_N:
						vf = cv2.VideoCapture(tfile.name)
						stframe = st.empty()

						while vf.isOpened():
							ret, frame = vf.read()
							if not ret:
								print("Can't receive frame (stream end?). Exiting ...")
								break
							frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
							image,label, conf =nudity_blur(frame, cfg_file_path, weights_file_path, names_file_path)
							stframe.image(image)
							if len(label) == 0:
								st.info('No Nudity present in the video')





		elif file_choice == 'Image':

			file=st.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png', 'webp', 'pdf'])

			if file is not None:
				if file.type != 'application/pdf':
					image = Image.open(file)
					image = np.array(image)

					st.sidebar.write('Select the required options')
					option_O = st.sidebar.checkbox('Original')
					option_E = st.sidebar.checkbox('Eyes')
					option_F = st.sidebar.checkbox('Face')
					option_S = st.sidebar.checkbox('Smile')
					option_N = st.sidebar.checkbox('Nudity')


					if st.button('Process'):

						if option_O and any([option_E, option_F, option_S, option_N]):
							st.warning('Cannot show Original and Masked image simultaneously')


						elif option_N and option_E and option_F and option_S:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image= face_blur(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_N and option_E and option_F:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image= face_blur(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_N and option_F and option_S:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image= face_blur(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_E and option_F and option_S:
							result_image= face_blur(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_N and option_E and option_S:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image= blur_smile(blur_eyes(result_image))
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)


						elif option_E and option_F:
							result_image= face_blur(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_S and option_F:
							result_image= face_blur(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)

						elif option_N and option_F:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image = face_blur(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
							if len(confidence) ==0:
								st.info('No nudity present in the image.')
							else:
								x=round(np.mean([float(i) for i in confidence])*100,2)
								st.info(f'Nudity percentage: {x}%')

						elif option_N and option_E:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image = blur_eyes(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
							if len(confidence) ==0:
								st.info('No nudity present in the image.')
							else:
								x=round(np.mean([float(i) for i in confidence])*100,2)
								st.info(f'Nudity percentage: {x}%')

						elif option_N and option_S:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							result_image = blur_smile(result_image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
							if len(confidence) ==0:
								st.info('No nudity present in the image.')
							else:
								x=round(np.mean([float(i) for i in confidence])*100,2)
								st.info(f'Nudity percentage: {x}%')

						elif option_E and option_S:
							result_image = blur_eyes(blur_smile(image))
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)



						elif option_O:
							result_image = image
							st.image(result_image, use_column_width=True)
							st.info(f'Oriinal image returned')
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
						elif option_E:
							result_image= blur_eyes(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
						elif option_F:
							result_image= face_blur(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
						elif option_S:
							result_image= blur_smile(image)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
						elif option_N:
							result_image, label, confidence= nudity_blur(image, cfg_file_path, weights_file_path, names_file_path)
							st.image(result_image, use_column_width=True)
							pil_img = Image.fromarray(result_image)
							st.markdown(get_image_download_link(pil_img), unsafe_allow_html=True)
						
							if len(confidence) ==0:
								st.info('No nudity present in the image.')
							else:
								x=round(np.mean([float(i) for i in confidence])*100,2)
								st.info(f'Nudity percentage: {x}%')


	elif choice =='About':
		about()

if __name__ == '__main__':
	main()


