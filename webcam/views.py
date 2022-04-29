from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import PIL
import cv2
import torch
import numpy as np
import datetime
import time

from PIL import Image
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# HOME PAGE -------------------------
def index(request):
	if request.method == 'POST':
		response = {}
		response['emotion_name'] = request.POST['emotion']
		return render(request, 'index.html', response)
	template = loader.get_template('index.html')
	return HttpResponse(template.render({}, request))

# -----------------------------------

def __zoom(img, scale=0.5, center=None): 
	height, width = img.shape[:2]

	if center is None: # 중심값이 초기값일 때의 계산 
		center_x = int(width / 2) 
		center_y = int(height / 2) 
		radius_x, radius_y = int(width / 2), int(height / 2) 
	else: # 특정 위치 지정시 계산 
		center_x, center_y = center 
		center_x, center_y = int(center_x), int(center_y) 
		left_x, right_x = center_x, int(width - center_x) 
		up_y, down_y = int(height - center_y), center_y 
		radius_x = min(left_x, right_x) 
		radius_y = min(up_y, down_y) 

	# 실제 zoom 코드 
	radius_x, radius_y = int(scale * radius_x), int(scale * radius_y) 
	# size 계산 
	min_x, max_x = center_x - radius_x, center_x + radius_x 
	min_y, max_y = center_y - radius_y, center_y + radius_y 
	
	# size에 맞춰 이미지를 자른다 
	cropped = img[min_y:max_y, min_x:max_x] 
	# 원래 사이즈로 늘려서 리턴 
	new_cropped = cv2.resize(cropped, (width, height)) 
	return new_cropped


# -----------------------------------

# DISPLAY CAMERA 1 ------------------
def stream_1(emotion_name):
	is_zoomin = False
	if emotion_name != 'none':
		is_zoomin = True

	print('[stream_1] is_zoomin : '+str(is_zoomin))

	cam_id = 0
	vid = cv2.VideoCapture(cam_id)

	while True:
		ret, frame = vid.read()
		#frame = cv2.resize(frame, (1000, 700))
		if(is_zoomin):
			frame = __zoom(frame)

		cv2.imwrite('currentframe.jpg', frame)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + open('currentframe.jpg', 'rb').read() + b'\r\n')
		
def video_feed_1(request, param):
	print('[video_feed_1] param : '+param)
	return StreamingHttpResponse(stream_1(param), content_type='multipart/x-mixed-replace; boundary=frame')
# -----------------------------------


'''

		#frame, class_count = detection(frame)

		#print("\nObjects in frame: ")
		row = 0

		#model = torch.hub.load('ultralytics/yolov3', 'custom', path='static/model/wider_total_v1.pt',force_reload=True)
	
		#results = model(frame).pandas().xyxy[0].values.tolist()
		#print(results)
		
		for k in range(len(class_count)):
			if class_count[k] > 0: 
				row += 1
				infor = str(obj_classes[k]) + ": " + str(int(class_count[k]))
				print("  " + infor)
				frame = cv2.putText(frame,infor,(20,(row+1)*35), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

# PARAMETERS FOR YOLO----------------
obj_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file         = "./yolov3_weight/yolov3_coco.pt"
num_classes     = 80
input_size      = 416
# -----------------------------------

def detection(vid):
	label_html = 'Capturing...'
	class_count = 0 
	model = torch.hub.load('ultralytics/yolov3', 'custom', path='static/model/wider_total_v1.pt',force_reload=True)
	#model = torch.hub.load('ultralytics/yolov3', 'custom',force_reload=True)
	#model.load_state_dict(torch.load('static/model/wider_total_v1.pt'))

	emotions = {'class0': '분노', 'class1': '기쁨', 'class2': '당황', 'class3': '중립', 'class4': '슬픔', 'class5': '불안','class6': '상처'}
	
	ret, frame = vid.read() 
	bbox_array = np.zeros([480,640,4], dtype=np.uint8)

	results = model(frame).pandas().xyxy[0].values.tolist()
	image2 = PIL.Image.fromarray(bbox_array)
	draw = PIL.ImageDraw.Draw(image2)
	for xmin, ymin, xmax, ymax, confidence, class_num, name in results:
		bbox_array = cv2.rectangle(bbox_array, (round(xmin), round(ymin)), (round(xmax), round(ymax)), (0, 0, 255), 2)
		bbox_array = cv2.putText(bbox_array, f"{emotions[name]} {float(confidence):0.3}",
							(round(xmin), round(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 0, 0), 2)
		class_count += 1
		bbox_array[:,:,3] = (bbox_array.max(axis = 2) > 0 ).astype(int) * 255
		# convert overlay of bbox into bytes
		#bbox_bytes = bbox_to_bytes(bbox_array)
		# update bbox so next frame gets new overlay
		#bbox = bbox_bytes
	return bbox_array, class_count


# YOLO DETECTION --------------------
def detection(vid):
	return_value, frame = vid.read()
	if return_value:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(frame)
	else:
		raise ValueError("No image!")


	frame_size = frame.shape[:2]
	image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
	image_data = image_data[np.newaxis, ...]
	prev_time = time.time()

	model = torch.hub.load('ultralytics/yolov3', 'custom', path='/content/drive/MyDrive/model/pro_wider_1000.pt', force_reload=True)

	pred_sbbox, pred_mbbox, pred_lbbox = model[]
	sess.run(
		[return_tensors[1], return_tensors[2], return_tensors[3]],
				feed_dict={ return_tensors[0]: image_data})

	pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
								np.reshape(pred_mbbox, (-1, 5 + num_classes)),
								np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

	bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
	bboxes = utils.nms(bboxes, 0.45, method='nms')
	image, detected = utils.draw_bbox(frame, bboxes)


	detected = np.asarray(detected)

	# print("------- frame i ---------")

	class_count = []

	for i in range(len(obj_classes)):   # 80
		obj_count = 0
		for j in range(len(detected)):
			if int(detected[j][5]) == i: obj_count += 1

		class_count = np.append(class_count, obj_count)

	curr_time = time.time()
	exec_time = curr_time - prev_time
	result = np.asarray(image)
	info = "time: %.2f ms" %(1000*exec_time)
	# cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
	result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
	return result, class_count
# -----------------------------------
'''
