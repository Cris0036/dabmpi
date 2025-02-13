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

import xml.etree.ElementTree as ET
from sympy import symbols, cos, sin, sqrt, lambdify, Interval, re, pi, S
import numpy as np
import mpmath

# Establecemos la precision de la integral a 16 decimales
mpmath.mp.dps = 16  


def read_config_from_xml(filename= '../data/param_config_magfield.xml'):
    tree = ET.parse(filename)
    root = tree.getroot()

    params = root.find("parameters")
    r_values = [float(r.text) for r in root.find("r_values")]

    return {
        "a": float(params.find("a").text),
        "beta": float(params.find("beta").text),
        "Bo": float(params.find("Bo").text),
        "Ro": float(params.find("Ro").text),
        "r_values": r_values,
    }


class ProblemCristina(object):
    def __init__(self, n_dimensions=32, config_file='../data/param_config_magfield.xml'):
        self.n_dimensions = n_dimensions

        # Leer los valores desde el archivo XML
        config = read_config_from_xml(config_file)
        self.a = config["a"]
        self.beta = config["beta"]
        self.Bo = config["Bo"]
        self.Ro = config["Ro"]
        self.r_values = config["r_values"]

        self.r, self.theta, self.q = symbols("r theta q")

        # Campo magnetico B = (Br, Btheta, Bphi)
        self.b_r = -self.Bo * self.beta * self.q * (((self.r**2) / (self.a**2)) - 1) * sin(self.theta) / 2
        self.b_theta = self.Bo * self.r / (self.Ro * self.q) + self.Bo * self.beta * self.q * cos(self.theta) * (3 * ((self.r**2) / (self.a**2)) - 1) / 2
        self.b_phi = self.Bo * (1 - self.r * cos(self.theta) / self.Ro - self.beta * (1 - self.r**2 / self.a**2) * (1 + self.beta * (self.q**2) * self.r * self.Ro * cos(self.theta) / (self.a**2)))

        self.modulo = sqrt(self.b_phi**2 + self.b_theta**2 + self.b_r**2)

        # Calculo de las derivadas parciales (gradiente del modulo de B)
        self.der_r = self.modulo.diff(self.r)
        self.der_theta = self.modulo.diff(self.theta)
        self.der_theta_grad = self.der_theta / self.r    # Gradiente en coordenadas toroidales = der_theta/r

        # B^3
        self.modulo3 = self.modulo **3

        # (B x grad(|B|))/ |B|^3 :
        self.x = - self.der_theta_grad * self.b_phi / self.modulo3
        self.y = self.der_r * self.b_phi / self.modulo3
        self.z = (self.der_theta_grad * self.b_r - self.der_r * self.b_theta) / self.modulo3

        self.modulo_final = sqrt(self.x**2 + self.y**2 + self.z**2)

        
        # Multiplicamos el jacobiano (= r*(Ro + r*cos(theta))) por el modulo final y nos sale el integrando,
        # tendriamos una integral doble con variables de integracion dtheta y dphi.
        # Al no depender el integrando de phi, se nos quedaría 2pi * (integral de 0 a 2pi
        # con el mismo integrando y dependiente de theta).

        # Podemos simplificar todavia mas, puesto que el resultado anterior hay que
        # dividirlo por el area del toroide (= 4*Ro*r*pi^2), se nos queda entonces:
        # (1/(2*Ro*pi))* (integral de 0 a 2pi de modulo_final*(Ro + r*cos(theta)) dtheta)
        
        # Siguiendo lo anterior, el integrando final se nos quedaria:
        self.integrando = self.modulo_final * (self.Ro + self.r * cos(self.theta))
        self.aux = 2 * np.pi * self.Ro
   
        
        
    def solve(self, solution):
        params = solution.getParametersValues()
       
        # Verificar que el numero de parametros sea igual al numero de dimensiones
        if len(params) != self.n_dimensions:
            raise ValueError(f"Expected {self.n_dimensions} parameters, but got {len(params)}")
            
        intervalo = Interval(0, 2*pi)
       
        val = 0 
        for i in range(len(params)): 
            r_val = self.r_values[i]
            q_val = params[i]
            
            # Evaluamos el integrando con los valores de r y q
            integrando_mp = lambdify(self.theta, self.integrando.subs({self.r: r_val, self.q: q_val}), modules={'mpmath': mpmath})
            # Calculamos la integral
            integral = mpmath.quad(integrando_mp, [0, 2*mpmath.pi])
            val += integral
            
        val = val / self.aux
        solution.setValue(val)
        return val

    def extractSolution(self):
        raise NotImplementedError("Extract solution abstract problem")

    def finish(self):
        raise NotImplementedError("Finish abstract problem")
