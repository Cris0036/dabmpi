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
    Version 0.1 (17-04-2013):   Creation of the file.
    Version 1.0 (15-01-2014):   Fist stable version.
"""

from SolutionBase import SolutionBase
from VMECData import VMECData
import Utils as u


class SolutionFusion (SolutionBase):
    def __init__(self, infile):
        SolutionBase.__init__(self, infile)
        self.__data = VMECData()
        self.__data.initialize(infile)
        return

    def initialize(self, data):
        self.__data = data
        return

    def prepare(self, filename):
        self.__data.create_input_file(filename)

    def getNumberofParams(self):
        return self.__data.getNumParams()

    def getMaxNumberofValues(self):
        return self.__data.getMaxRange()

    def getParameters(self):
        return self.__data.getParameters()

    def getParametersValues(self):
        return self.__data.getValsOfParameters()

    """
    Receives an array of floats and sets the values of the parameters
    """

    def setParametersValues(self, buff):
        u.logger.debug("SolutionFusion. Setting parameters")
        self.__data.setValsOfParameters(buff)

    #Receives a list of parameters with, at least, index and value
    #Updates the parameters of the object with the new values specified in
    #the list
    def setParameters(self, params):
        for p in params:
            self.__data.assign_parameter(p)

    def getData(self):
        return self.__data

    def checkPressureDerivative(self):
        values = self.getParametersValues()
        val = 0.0
        for v in range(-100, 100):
            for i in range(len(values)):
                if (i == 0):
                    continue
                val += pow(v, i - 1) * values[i] * i
        if (val > 0.0):
            return False
        return True
