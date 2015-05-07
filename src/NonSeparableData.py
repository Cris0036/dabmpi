#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

#############################################################################
#    Copyright 2013  by Antonio Gomez and Miguel Cardenas                   #
#                                                                           #
#   Licensed under the Apache License, Version 2.0 (the "License");         #
#   you may not use this file except in compliance with the License.        #
#   You may obtain a copy of the License at                                 #
#                                                                           #
#       http://www.apache.org/licenses/LICENSE-2.0                          #
#                                                                           #
#   Unless required by applicable law or agreed to in writing, software     #
#   distributed under the License is distributed on an "AS IS" BASIS,       #
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.#
#   See the License for the specific language governing permissions and     #
#   limitations under the License.                                          #
#############################################################################

__author__ = ' AUTHORS:     Antonio Gomez (antonio.gomez@csiro.au)'


__version__ = ' REVISION:   1.0  -  15-01-2014'

"""
HISTORY
    Version 0.1 (12-04-2013):   Creation of the file.
    Version 1.0 (15-01-2014):   Fist stable version.
"""
import os
import sys
import traceback
from xml.dom import minidom
import Utils as u
from Parameter import Parameter
from array import array


class NonSeparableData (object):
    """
    This class stores all the data required by VMEC.
    It also provides methods to read the input xml file that, for each
    parameter, specifies the min/max/default values, the gap, the index,...
    It can also create an XML output file with the data it contains
    """
    def __init__(self):
        self.__numParams = 0
        self.__maxRange = 0
        self.__fInput = None
        self.__params = []
        return

    def __del__(self):
        try:
            del self.__params[:]
        except:
            pass

    #returns the number of parameters that can be actually modified
    def getNumParams(self):
        return self.__numParams

    """
    Returns a list of doubles with the values of the modificable parameters
    """
    def getValsOfParameters(self):
        buff = array('f', [0]) * self.__numParams
        for i in range(self.__numParams):
            buff[i] = float(self.__params[i].get_value())
        return buff

    """
    Receives as parameter (buff) a list of values corresponding to
    the values that the parameters that can be modified must take.
    Goes through all of the parameters and changes the values of the
    modificable parameters to the value specified in this list
    """

    def setValsOfParameters(self, buff):
        u.logger.debug("NonSeparableData. Setting parameters (number: "
                         + str(len(buff)) + ")")
        for i in range(len(buff)):
            self.__params[i].set_value(buff[i])

    def setParameters(self, parameters):
        for param in parameters:
            self.assign_parameter(param)
    """
    Return a list with all the parameters (list of Parameter objects)
    """
    def getParameters(self):
        return self.__params

    """
    Assign a parameter with a new value to an older version of the
    same parameter
    """
    def assign_parameter(self, parameter):
        index = parameter.get_index()
        if (index >= len(self.__params)):
            self.__params.append(parameter)

    def getMaxRange(self):
        return self.__maxRange

    """
    Method that reads the xml input file. Puts into memory all the data
    contained in that file (min-max values, initial value, gap,...)
    Argument:
        - filepath: path to the XML input file
    """

    def initialize(self, filepath):
        try:
            if (not os.path.exists(filepath)):
                filepath = "../" + filepath
            xmldoc = minidom.parse(filepath)
            pNode = xmldoc.childNodes[0]
            for node in pNode.childNodes:
                if node.nodeType == node.ELEMENT_NODE:
                    c = Parameter()
                    for node_param in node.childNodes:
                        if (node_param.nodeType == node_param.ELEMENT_NODE):
                            if (node_param.localName == "type"):
                                c.set_type(node_param.firstChild.data)
                            if (node_param.localName == "value"):
                                c.set_value(node_param.firstChild.data)
                            if (node_param.localName == "min_value"):
                                c.set_min_value(node_param.firstChild.data)
                            if (node_param.localName == "max_value"):
                                c.set_max_value(node_param.firstChild.data)
                            if (node_param.localName == "index"):
                                c.set_index(node_param.firstChild.data)
                            if (node_param.localName == "name"):
                                c.set_name(node_param.firstChild.data)
                            if (node_param.localName == "gap"):
                                c.set_gap(node_param.firstChild.data)
                    try:
                        self.assign_parameter(c)
                        self.__numParams += 1
                        values = 1 + int(round((c.get_max_value() -
                                c.get_min_value()) / c.get_gap()))
                        self.__maxRange = max(values, self.__maxRange)
                    except Exception, e:
                        traceback.print_tb(sys.exc_info()[2])
                        u.logger.warning("Problem calculating max range: " +
                                         str(e))
                        pass
            u.logger.debug("Number of parameters " +
                           str(self.__numParams) + "(" +
                           str(len(self.__params)) + ")")
        except Exception, e:
            u.logger.error("NonSeparableData (" +
                            str(sys.exc_traceback.tb_lineno) +
                            "). Problem reading input xml file: " + str(e))
            traceback.print_tb(sys.exc_info()[2])
            sys.exit(111)
        return
