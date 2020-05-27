"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def bounding_box(frame, result, prob_threshold, initial_w, initial_h):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if  obj[2] >= prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 1)
            #Intializing aboove program using numpy array
            # xmin,ymin,xmax,ymax= obj[3:7] * np.array([initial_w, initial_h, initial_w, initial_h])
            current_count = current_count + 1
    return frame, current_count

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.
    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    prob_threshold, initial_w, initial_h
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    model = args.model
    video = args.input
    request = 0
    Device = args.device
    Cpu_extension = args.cpu_extension
    single_img_flag = False
    

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(model, Device,Cpu_extension)
    network_shape = infer_network.get_input_shape()
    ### TODO: Handle the input stream ###
    #check for live video cam
    if video == 'CAM': 
        input_validation = 0
        
        # Checks for input image


    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_flag = True
        input_validation = video

    # Checks for video file
    else:
        input_validation = video
        assert os.path.isfile(args.input), "file doesn't exist"	
    ### TODO: Handle the input stream ###
    
    cap = cv2.VideoCapture(input_validation)
    cap.open(input_validation)

    initial_w = int(cap.get(3))
    initial_h = int(cap.get(4))

   # in_shape = network_shape['image_tensor']

    #iniatilize variables
    
    duration = 0
    last_duration=0
    counter=0
    total_count = 0
    dur = 0
    report = 0
    current_count = 0
    last_count = 0
    infer_time=0
    total_frames=0
    ### TODO: Loop until stream is over ###
    
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        prob_threshold = args.prob_threshold
        if not flag:
            break
  
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        image = cv2.resize(frame, (network_shape[3],network_shape[2]))
        image_p = image.transpose((2, 0, 1))
        image_p = image_p.reshape(1, *image_p.shape)

        ### TODO: Start asynchronous inference for specified request ###
       # net_input = {'image_tensor': image_p,'image_info': image_p.shape[1:]}
        duration_report = None
        inf_start = time.time()
        infer_network.exec_net(request, image_p)
        color = (255,0,0)


        ### TODO: Wait for the result ###
        if infer_network.wait(request) == 0:
            det_time = time.time() - inf_start
            infer_time=infer_time + det_time * 1000
            total_frames = total_frames +1
            # Results of the output layer of the network
            result = infer_network.get_output(request)
            #if args.perf_counts:
                #perf_count = infer_network.performance_counter(request)
                #performance_counts(perf_count)

            frame, current_count = bounding_box(frame, result, initial_w, initial_h)
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 15),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (0, 55, 255), 1)
            message = "Average time: {:.3f}ms".format(infer_time/total_frames)
            cv2.putText(frame, message, (15, 35), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255,0,0), 1)
            ## Delaying the detection/undetection by 15 duration frame to remove issues with wrongly calculated durations and counts due to flickering
            if current_count!=counter:
                last_count=counter
                counter=current_count
            
                if duration>=15:
                    last_duration=duration
                    duration=0
                else:
                    duration=last_duration+duration
                    last_duration=0
            else:
                duration=duration+1
                if duration>=15:
                    report==counter
                    ###todo calculate and send relevant information on###
                    # When new person enters the video
                    if duration==15 and counter>last_count:  
                        total_count=total_count+counter-last_count
                        client.publish("person", json.dumps({"count":report, "total": total_count}))
                        # When new person leaves the video
                    elif duration==15 and counter<last_count:
                        duration_report=int((last_duration/10.0)*1000)
                        client.publish("person/duration", json.dumps({"duration": duration_report}))
                #adding current count to frame 
                dist = "current_count: %d " %current_count
                cv2.putText(frame, dist, (15, 65), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255,0,0), 1)
                client.publish("person", json.dumps({"count": current_count}))
            
             #adding total count to frame 
            tot = "total_count: %d " %total_count
            cv2.putText(frame, tot, (15, 85), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.5, (255,0,0), 1)

              
            
            

        ### TODO: Send the frame to the FFMPEG server ###
        #frame = cv2.resize(frame, (768, 432))
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()
        if key_pressed == 27:
               #print ("Exiting due to keyboard interrupt");
                break

        ### TODO: Write an output image if `single_image_mode` ###
        if single_img_flag:
            cv2.imwrite('output_image.jpg', frame)
        ### TODO: Close the stream and any windows at the end of the application
        #cap.release()
        #cap.release()
        #cv2.destroyAllWindows()



def main():
    """
    Load the network and parse the output.
    :return: None
    """
    # Grab command line args
    args = build_argparser()
    args =args.parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
