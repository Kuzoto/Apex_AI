from roboflow import Roboflow
import supervision as sv
import cv2
import pyautogui
from pynput.keyboard import Key, Listener
import time
import numpy as np
import mss
import inference
from PIL import Image

# rf = Roboflow(api_key="wybH8DeH7sYNwpm8gXoB")
# project = rf.workspace().project("apex-ai-2-khlrg")
# model = project.version(1).model
# model2 = project.version(2).model

model = inference.get_model(model_id="apex-ai-2-khlrg/1", api_key="wybH8DeH7sYNwpm8gXoB", onnxruntime_execution_providers=["CUDAExecutionProvider"])
model2 = inference.get_model(model_id="merged-apex-ai-mb4ez/1", api_key="wybH8DeH7sYNwpm8gXoB", onnxruntime_execution_providers=["CUDAExecutionProvider"])


# def on_press(key):
#     print('{0} pressed'.format(
#         key))
#     if key == Key.f5:
#         # Stop listener
#         listener.stop()

# listener = Listener(on_press=on_press)
# listener.start()


while True:
    with mss.mss() as sct:
        full_screenshot = sct.grab(sct.monitors[1]) # sct.monitors[1] is the primary display

        # Convert the captured data to a Pillow Image
        img = Image.frombytes("RGB", full_screenshot.size, full_screenshot.rgb)

        # Resize the image to your desired resolution
        # Example: Resize to half the original dimensions
        new_width = int(img.width / 2)
        new_height = int(img.height / 2)
        resized_img = img.resize((new_width, new_height))

        # Save the resized image
        resized_img.save("temp_image.jpg")
    #screenshot = mss.mss().shot(mon=2, output="temp_image.jpg")
    #print("screenshot")
    #screenshot = pyautogui.screenshot()
    # result = model.predict(screenshot, confidence=40, overlap=50).json()
    # result2 = model2.predict(screenshot, confidence=40, overlap=50).json()
    result = model.infer(img)[0]
    result2 = model2.infer(img)[0]
    #print("predicted")
    # labels = [item["class"] for item in result["predictions"]] + [item["class"] for item in result2["predictions"]]
    # labels = set(labels)
    # labels = list(labels)

    # print(labels)

    detections = [sv.Detections.from_inference(result), sv.Detections.from_inference(result2)]
    #print("detect")
    detections = sv.Detections.merge(detections)
    detections = detections.with_nmm(threshold=0.50)
    #print("merge")
    if "class_name" in detections.data:
        labels = detections.data['class_name']
        #print("labels")
        #print(labels)

        label_annotator = sv.LabelAnnotator()
        box_annotator = sv.BoxAnnotator()

        image = cv2.imread("temp_image.jpg")
        #image = screenshot
        #print("read")

        annotated_image = box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)
        #print("annotated")
        sv.ImageSink(target_dir_path=".").save_image(image=annotated_image, image_name="anno_image.jpg")
        #print("saved")
    # with Listener(
    #         on_press=on_press,
    #         on_release=on_release) as listener:
    #         listener.join(timeout=0.025)
    #time.sleep(0.0083)
    #print("slept")
    break
# listener.stop()
#sv.plot_image(image=annotated_image, size=(16, 16))