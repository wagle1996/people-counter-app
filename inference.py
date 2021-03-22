#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        self.plugin=IECore()
        ### TODO: Load the model ###
        ### Load the Inference Engine API
        self.plugin = IECore()

    ### Load IR files into their related class
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.network = IENetwork(model=model_xml, weights=model_bin)

    ### Add a CPU extension, if applicable.
        if os.path.isfile(cpu_extension)  and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)

    ### Get the supported layers of the network
        supported_layers_path = self.plugin.query_network(network=self.network, device_name="CPU")

    ### Check for any unsupported layers, and let the user know if anything is missing. Exit the program, if so.
        keys=self.network.layers.keys()
        for l in keys:
            unsupported_layers_path=""
            if l not in supported_layers_path:
                unsupported_layers_path=l
        if len(unsupported_layers_path) != 0:
            #print("Unsupported layers found: {}".format(unsupported_layers_path))
            #print("Check whether the extensions are available to add to IECore.")
            exit(1) 
    ### Load the network into the Inference Engine
        self.exec_network=self.plugin.load_network(self.network, "CPU")
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        #print("IR is successfully loaded into Inference Engine.")
        #print

        return
        ### Note: You may need to update the function parameters. ###

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        #it returns the shape of the input layer
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, request_id,image):
        #start acynchronous request
        self.infer_request_handle = self.exec_network.start_async(request_id= request_id, inputs={self.input_blob: image})
        return self.exec_network
    def wait(self,  request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        wait_n = self.exec_network.requests[request_id].wait(-1)
        return wait_n


    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[request_id].outputs[self.output_blob]
