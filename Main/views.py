import os
import cv2
import json
import time
import torch
import numpy as np
from gtts import gTTS
from django.conf import settings
from pycocotools.coco import COCO
from googletrans import Translator
from django.shortcuts import render
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from django.core.files.storage import FileSystemStorage

# Create your views here.

translator = Translator()

def load_coco_categories():
    annotations_file_path = os.path.join(settings.MEDIA_ROOT, 'coco_annotations/instances_train2017.json')
    with open(annotations_file_path, 'r') as f:
        data = json.load(f)
    category_map = {category['id']: category['name'] for category in data['categories']}
    return category_map

def get_prediction(img_path, threshold=0.5):
    img = Image.open(img_path)
    transform = transforms.Compose([transforms.ToTensor()])
    img = transform(img)
    
    with torch.no_grad():
        pred = settings.MASK_RCNN_MODEL([img])
    
    pred_scores = pred[0]['scores'].numpy()
    pred_boxes = pred[0]['boxes'].detach().numpy()
    pred_labels = pred[0]['labels'].numpy()

    pred_thresh_indices = np.where(pred_scores >= threshold)[0]
    pred_boxes = pred_boxes[pred_thresh_indices]
    pred_labels = pred_labels[pred_thresh_indices]
    
    return pred_boxes, pred_labels, pred_scores[pred_thresh_indices]

def get_translated_label(label):
    translation = translator.translate(label, src='en', dest='hi')
    translated_label = translation.text
    return translated_label

def draw_bounding_boxes(img_path, boxes, labels, coco_categories):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    draw = ImageDraw.Draw(pil_img)
    font_path = "C:/Windows/Fonts/mangal.ttf"
    font = ImageFont.truetype(font_path, 40)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(i) for i in box]
        label = coco_categories[labels[i]]
        translated_label = get_translated_label(label)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 20), translated_label, font=font, fill=(36, 255, 12))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def draw_yolov8_bounding_boxes(img_path, boxes, labels, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    draw = ImageDraw.Draw(pil_img)
    font_path = "C:/Windows/Fonts/mangal.ttf"
    font = ImageFont.truetype(font_path, 40)

    class_names = model.names
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        label = class_names[labels[i]]
        translated_label = get_translated_label(label)
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 20), translated_label, font=font, fill=(36, 255, 12))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img

def index_view(request):
    if request.method == 'POST':
        start_time = time.time()
        image_file = request.FILES['file']
        fs = FileSystemStorage()
        file_path = fs.save(image_file.name, image_file)
        file_url = fs.url(file_path)

        file_full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        if request.POST.get('selected_model') == 'RCNN':

            coco_categories = load_coco_categories()
            boxes, labels, scores = get_prediction(file_full_path)
            result_image = draw_bounding_boxes(file_full_path, boxes, labels, coco_categories)
            result_img_path = file_full_path.replace(".jpg", "_result.jpg")
            cv2.imwrite(result_img_path, result_image)

            result_img_url = file_url.replace(".jpg", "_result.jpg")
            detected_labels_en = [coco_categories[label] for label in labels]
            detected_labels_hi = [get_translated_label(coco_categories[label]) for label in labels]
            scores_hi = [get_translated_label(str(score)) for score in scores]
            detected_objects_en = [{'label': label, 'score': score} for label, score in zip(detected_labels_en, scores)]
            detected_objects_hi = [{'label': label, 'score': score} for label, score in zip(detected_labels_hi, scores_hi)]
            model_name = 'Mask RCNN'

        else:

            results = settings.YOLO_V8_MODEL(file_full_path)
            boxes, labels, scores = [], [], []
            for result in results:
                for box in result.boxes.xyxy:
                    boxes.append(box.cpu().numpy())
                labels.extend(result.boxes.cls.cpu().numpy().astype(int))
                scores.extend(result.boxes.conf.cpu().numpy())

            result_image = draw_yolov8_bounding_boxes(file_full_path, boxes, labels, settings.YOLO_V8_MODEL)
            result_img_path = file_full_path.replace(".jpg", "_result.jpg")
            cv2.imwrite(result_img_path, result_image)

            result_img_url = file_url.replace(".jpg", "_result.jpg")
            detected_labels_en = [settings.YOLO_V8_MODEL.names[label] for label in labels]
            detected_labels_hi = [get_translated_label(settings.YOLO_V8_MODEL.names[label]) for label in labels]
            scores_hi = [get_translated_label(str(score)) for score in scores]
            detected_objects_en = [{'label': label, 'score': score} for label, score in zip(detected_labels_en, scores)]
            detected_objects_hi = [{'label': label, 'score': score} for label, score in zip(detected_labels_hi, scores_hi)]
            model_name = 'YOLOv8'
            
        labels_string_en = ', '.join(detected_labels_en)
        en_audio = gTTS(text=f'Detected object {labels_string_en}', lang='en')

        labels_string_hi = ', '.join(detected_labels_hi)
        hi_audio = gTTS(text=f'पता लगाई गई वस्तुएँ {labels_string_hi} है', lang='hi')

        en_audio_filename = f'en_audio.mp3'
        audio_path = os.path.join(settings.MEDIA_ROOT, en_audio_filename)
        en_audio.save(audio_path)

        hi_audio_filename = f'hi_audio.mp3'
        audio_path = os.path.join(settings.MEDIA_ROOT, hi_audio_filename)
        hi_audio.save(audio_path)

        en_audio_url = os.path.join(settings.MEDIA_URL, en_audio_filename)
        hi_audio_url = os.path.join(settings.MEDIA_URL, hi_audio_filename)
        end_time = time.time()
        elapsed_time = end_time - start_time
        context = {
            'result':True,
            'model':model_name,
            'result_image_url':result_img_url,
            'detected_objects_en':detected_objects_en,
            'detected_objects_hi':detected_objects_hi,
            'en_audio_url': en_audio_url, 
            'hi_audio_url': hi_audio_url,
            'elapsed_time': elapsed_time,
        }
        return render(request, 'index.html', context=context)
    else:
        return render(request, 'index.html')