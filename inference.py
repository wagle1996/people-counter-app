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
from openvino.inference_engine import IENetwork, IECore,IEPlugin


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec = None
        self.infer_request = None

    def load_model(self, model, device, input_size, output_size, num_requests, cpu_extension=None, plugin=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Plugin initialization for the required device
        # load extension libraries
        if not plugin:
            log.info("Initializing plugin for {} device...".format(device))
            self.plugin = IEPlugin(device=device)
        else:
            self.plugin = plugin
            
        #check cpu if required

        if cpu_extension and 'CPU' in device:
            self.plugin.add_cpu_extension(cpu_extension)

        # Read Intermediate representation 
        log.info("Reading IR...")
        self.network = IENetwork(model=model_xml, weights=model_bin)
        log.info("Loading IR to the plugin...")

        if self.plugin.device == "CPU":
            supported_layers = self.plugin.get_supported_layers(self.network)
            not_supported_layers = \
                [l for l in self.network.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by "
                          "the plugin for specified device {}:\n {}".
                          format(self.plugin.device,
                                 ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path"
                          " in command line parameters using -l "
                          "or --cpu_extension command line argument")
                sys.exit(1)

        if num_requests == 0:
            # Loads network read from IR to the plugin
            self.exec = self.plugin.load(network=self.network)
        else:
            self.exec = self.plugin.load(network=self.network, num_requests=num_requests)

        self.input_blob = next(iter(self.network.inputs))
        self.out_blob = next(iter(self.network.outputs))
        assert len(self.network.inputs.keys()) == input_size, \
            "Supports only {} input topologies".format(len(self.network.inputs))
        assert len(self.network.outputs) == output_size, \
            "Supports only {} output topologies".format(len(self.network.outputs))

        return self.plugin, self.get_input_shape()

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        #it returns the shape of the unput layer
        return self.network.inputs[self.input_blob].shape
        

    def exec_net(self, cur_request_id,image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.infer_request_handle = self.exec.start_async(
            request_id=cur_request_id, inputs={self.input_blob: image})
        return self.exec

    def wait(self,cur_request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        wait_n = self.exec.requests[cur_request_id].wait(-1)
        return wait_n


    def get_output(self, request_id, output=None):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ##
        if output:
            result = self.infer_request_handle.outputs[output]
        else:
            result = self.exec.requests[request_id].outputs[self.out_blob]
        return result
