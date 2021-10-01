import numpy as N
import pylab as P

def rhs(t,y):
    A =  N.array([[0,1], [-2,-1]])
    yd = N.dot(A,y)

    return yd

y0 = N.array([1.0,1.0])
t0 = 0.0

from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVode

model = Explicit_Problem(rhs, y0, t0)
model.name = 'Linear Test ODE'

sim = CVode(model)

tfinal = 10.0

t, y = sim.simulate(tfinal)

P.plot(t,y)
P.savefig('sim_result.png')

import pdb; pdb.set_trace()
