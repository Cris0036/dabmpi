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

#import random

#class ProblemCristina(object):
 #   def __init__(self):
  #      return

   # def solve(self, solution):
    #    val = random.randint(0, 1000000)
     #   solution.setValue(val)
      #  print("ProblemCristina. Solution found with value: " + str(val))
       # return val

    #def extractSolution(self):
     #   raise NotImplementedError("Extract solution abstract problem")

    #def finish(self):
     #   raise NotImplementedError("Finish abstract problem")

import numpy as np

class ProblemCristina(object):
    def __init__(self, n_dimensions=1):
        # Numero dimensiones para la funcion de Schwefel 2.22
        self.n_dimensions = n_dimensions

    def solve(self, solution):
        params = solution.getParametersValues()

        
        # Check number parameters == n_dimensions
        if len(params) != self.n_dimensions:
            raise ValueError(f"Expected {self.n_dimensions} parameters, but got {len(params)}")

        # Funcion de Schwefel 2.22
        # val = sum([abs(x) for x in params]) + np.prod([abs(x) for x in params])
        
        # Otra funcion
        
        # val = abs(params[0]) + abs(params[1]) + 14
        
        
        # PRUEBA GRADO 4
        val = params[0]**4 - 3*params[0]**3 + 2*params[0]**2 + 5
       
        solution.setValue(val)
        print("ProblemCristina. Solution found with value: " + str(val), "parameters = ", params)
        
        return val

    def extractSolution(self):
        raise NotImplementedError("Extract solution abstract problem")

    def finish(self):
        raise NotImplementedError("Finish abstract problem")
