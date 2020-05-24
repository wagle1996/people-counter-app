# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers
I didnot used custom layers while doing my project.

## Model conversion process

 First of all i used wget to download the model from the repository via link http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz. It is Supported Frozen Topology from TensorFlow Object Detection Models Zoo. after that i unzipped the tar using Tar -xvf command. 
using tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz


after that for converting model to IR I  downloaded SSD MobileNet V2 COCO model's .pb file using the model optimizer using the command.
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

In above command opt/intel/openvino/deployment_tools/model_optimizer/mo.py path is path to model optimizer file. The next argument input_model frozen_inference_graph.pb is the input model for conversion which is in pb format. next is configuration file . this is pipe line config file. I also reveresed the input channel. At last fed in the json file.the conversion was sucessful, it formed .xml file and .bin file. 

The Generated IR model files are :

XML file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.xml
BIN file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.bin

Finally i run the program using command:
for video file: python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m "/home/workspace/intel//ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


I also converted tensorflow-yolo-v3 model to Inference engine. For that i cloned git repository of yolo v3 model using link https://github.com/mystic123/tensorflow-yolo-v3.git after that downloaded the coco_names and yolov3.weights. And finally run the converter using command
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
and finally run the model using python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m "/home/workspace/tensorflow-yolo-v3/frozen_darknet_yolov3_model.xml" -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were :

## SSD_mobilenet_v2_coco 
The size of ssd_mobilenet_v2_coco the model pre- and post-conversion was almost the same. The SSD MobileNet V2 COCO model .pb file is about 66.4 MB and the IR bin file is 64.1 MB. The performance of this model was not good. When there were 2 person on the frame the total people counted was 6.
The inference time of the model pre- and post-conversion was approximately about 73ms. It's mean accuracy precision was 21 map. The accuracy was obtained from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md.

## tensorflow_yolo_v3
The size of yolo_v3 after conversion was heavier than earlier one. The Size of frozen_darknet_yolo.pb was 237 mb and size of frozen_darknet_yolo.bin was 236 mb. The performance of this model was worst among others. The inference time was fluctuating heavily. But the average time was 1095 ms most of the times. the map was 33 map. This model didnot detected person. 


## Pretrained person detection Model
I also  tested pretrained model from open zoo and found that pretained model has less inference time than the open source model. The pretrained model i used was person-detection-retail-0013 model. It had inference time of 43ms only. It had better accuracy than the previous. The size of the model file was only 2.59 mb.



## Assess Model Use Cases

Some of the potential use cases of the people counter app are, at the shops to keep the track of cutomers by their interest, and at the traffic signal to make sure that people crosses safely.Monitor passenger traffic flow in air port and train station and Assign staff deployment based on demand. It is also very useful in queue management. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

-In day light the quality is not affected but in the night light and dim light the performance is affected.

-Camera angle also highly affects the accuracy. The model did not detected properly when the camera angle was changed

-The image size should also be in proporational with the model accuracy. Otherwise the detection confusion matrix will be highly affected. There will be more false negatives.

