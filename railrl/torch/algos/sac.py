"""
Soft actor-critic

Notes from Tuomas
a = tanh(z)
Q(a, s) = Q(tanh(z), s)
z is output of actor

Policy
pi ~ e^{Q(s, a)}
   ~ e^{Q(s, tanh(z))}

Mean
 - regularize gaussian means of policy towards zeros

Covariance
 - output log of diagonals
 - Clip the log of the diagonals
 - regularize the log towards zero
"""