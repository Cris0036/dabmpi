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

from sympy import symbols, cos, sin, sqrt, lambdify, singularities, Interval, re, pi, S
import numpy as np
import mpmath

# Establecemos la precision de la integral a 16 decimales
mpmath.mp.dps = 16  

class ProblemCristina(object):
    def __init__(self, n_dimensions=32):
        self.n_dimensions = n_dimensions
        self.r_values = [
            0.06060606, 0.12121212, 0.18181818, 0.24242424, 0.3030303 ,
            0.36363636, 0.42424242, 0.48484848, 0.54545455, 0.60606061,
            0.66666667, 0.72727273, 0.78787879, 0.84848485, 0.90909091,
            0.96969697, 1.03030303, 1.09090909, 1.15151515, 1.21212121,
            1.27272727, 1.33333333, 1.39393939, 1.45454545, 1.51515152,
            1.57575758, 1.63636364, 1.6969697 , 1.75757576, 1.81818182,
            1.87878788, 1.93939394
        ]

        self.r, self.theta, self.q = symbols('r theta q')
        
        # Configuracion del ITER para el campo magnetico
        self.a = 2.0
        self.beta = 1.8
        self.Bo = 5.3
        self.Ro = 6.2

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
        # tendríamos una integral doble con variables de integracion dtheta y dphi.
        # Al no depender el integrando de phi, se nos quedaría 2pi * (integral de 0 a 2pi
        # con el mismo integrando y dependiente de theta).

        # Podemos simplificar todavía más, puesto que el resultado anterior hay que
        # dividirlo por el area del toroide (= 4*Ro*r*pi^2), se nos queda entonces:
        # (1/(2*Ro*pi))* (integral de 0 a 2pi de modulo_final*(Ro + r*cos(theta)) dtheta)
        
        # Siguiendo lo anterior, el integrando final se nos quedaría:
        self.integrando = self.modulo_final * (self.Ro + self.r * cos(self.theta))
        self.aux = 2 * np.pi * self.Ro
   
        
        
    def solve(self, solution):
        params = solution.getParametersValues()
       
        # Verificar que el número de parámetros sea igual al número de dimensiones
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
