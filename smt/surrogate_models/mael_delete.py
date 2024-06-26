import matplotlib.pyplot as plt
import numpy as np
from smt.surrogate_models.krg import KRG

from smt.surrogate_models import KRG

xt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
yt = np.array([0.0, 1.0, 1.5, 0.9, 1.0])
num = 100

x = np.linspace(0.0, 4.0, num)
##### Normal
# sm = KRG()
# sm.set_training_values(xt, yt)
# sm.train()
# y = sm.predict_values(x)
# y_var = sm.predict_variances

##### Noisy
sm_noisy = KRG(noise0=[1e-1])
sm_noisy.set_training_values(xt, yt)
sm_noisy.train()
y_noisy = sm_noisy.predict_values(x)
# print(y_noisy)


# ##### estimated y
# print("normal", y[:10])
# print("noisy", y_noisy[:10])
# print(np.allclose(y, y_noisy, rtol=0.1))

##### estimated variance
var = sm_noisy.predict_variances(xt, is_ri=False)
var_noisy = sm_noisy.predict_variances(xt, is_ri=True)  # Je ne comprends pas pourquoi sigma_ri = None TODO
print("variance normale au training", var)
print("variance noisy au training", var_noisy)
