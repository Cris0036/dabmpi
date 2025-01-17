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
    def __init__(self, n_dimensions=1):
        self.n_dimensions = n_dimensions
        self.r_values = [
            0.2, 0.04081633, 0.08163265, 0.12244898, 0.16326531,
            0.20408163, 0.24489796, 0.28571429, 0.32653061, 0.36734694,
            0.40816327, 0.44897959, 0.48979592, 0.53061224, 0.57142857,
            0.6122449 , 0.65306122, 0.69387755, 0.73469388, 0.7755102 ,
            0.81632653, 0.85714286, 0.89795918, 0.93877551, 0.97959184,
            1.02040816, 1.06122449, 1.10204082, 1.14285714, 1.18367347,
            1.2244898 , 1.26530612, 1.30612245, 1.34693878, 1.3877551 ,
            1.42857143, 1.46938776, 1.51020408, 1.55102041, 1.59183673,
            1.63265306, 1.67346939, 1.71428571, 1.75510204, 1.79591837,
            1.83673469, 1.87755102, 1.91836735, 1.95918367, 1.9999
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
        print("modulo_final terminado")
        
        # Multiplicamos el jacobiano (= r*(Ro + r*cos(theta))) por el modulo final y nos sale el integrando,
        # tendríamos una integral doble con variables de integracion dtheta y dphi.
        # Al no depender el integrando de phi, se nos quedaría 2pi * (integral de 0 a 2pi
        # con el mismo integrando y dependiente de theta).

        # Podemos simplificar todavía más, puesto que el resultado anterior hay que
        # dividirlo por el area del toroide (= 4*Ro*r*pi^2), se nos queda entonces:
        # (1/(2*Ro*pi))* (integral de 0 a 2pi de modulo_final*(Ro + r*cos(theta)) dtheta)
        
        # Siguiendo lo anterior, el integrando final se nos quedaría:
        self.integrando = self.modulo_final * (self.Ro + self.r * cos(self.theta))
        print("ok")
        # self.integrando_func = lambdify((self.theta, self.r, self.q), self.integrando, modules={'numpy': np})
        # print("ok2")
        self.aux = 2 * np.pi * self.Ro
        print("init bien terminado")
        
        
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
        
            # Encontrar las singularidades en el intervalo [0,2pi]
            singularidades = singularities(self.integrando.subs({self.r: r_val, self.q: q_val}), self.theta, domain=intervalo)
            singularidades_reales = [float(re(s.evalf())) for s in singularidades if s.is_real]
            if singularidades != S.EmptySet:
                print(singularidades)
                print("Singularidades reales encontradas en theta:", singularidades_reales, r_val, q_val)
            integral = mpmath.quad(integrando_mp, [0, 2*mpmath.pi], points=singularidades_reales)
            val += integral
        
            
        val = val / self.aux
        solution.setValue(val)
        return val

    def extractSolution(self):
        raise NotImplementedError("Extract solution abstract problem")

    def finish(self):
        raise NotImplementedError("Finish abstract problem")
