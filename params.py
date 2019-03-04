from full_model import run_model, run_ae, run_ssc
from skopt.space import Integer, Real

all_params = [
  # 0
  {'model':run_model, 'dataset':'yaleB', 'n_rand':10, 'epochs_pretrain':201, 'epochs':101, 'space':
       [Real(10**-2, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-3, 10**-1, "log-uniform", name='lr'),
        Real(10**0, 10**2, "log-uniform", name='alpha1'),
        Integer(2, 32, name='maxIter1'),
        Real(10**0, 10**2, "log-uniform", name='alpha2'),
        Integer(2, 32, name='maxIter2')]},
  {'model':run_model, 'dataset':'yaleB', 'n_rand':10, 'epochs_pretrain':201, 'epochs':101, 'space':
       [Real(10**-2, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-3, 10**-1, "log-uniform", name='lr'),
        Real(10**0, 10**2, "log-uniform", name='alpha1'),
        Integer(10, 32, name='maxIter1'),
        Real(10**0, 10**2, "log-uniform", name='alpha2'),
        Integer(10, 32, name='maxIter2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':201, 'epochs':101, 'space':
       [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-5, 10**-1, "log-uniform", name='lr'),
        Real(10**-1, 10**3, "log-uniform", name='alpha1'),
        Integer(10, 100, name='maxIter1'),
        Real(10**-1, 10**3, "log-uniform", name='alpha2'),
        Integer(10, 100, name='maxIter2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':1001, 'epochs':251, 'space':
       [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-5, 10**-1, "log-uniform", name='lr'),
        Real(10**-1, 10**3, "log-uniform", name='alpha1'),
        Integer(10, 200, name='maxIter1'),
        Real(10**-1, 10**3, "log-uniform", name='alpha2'),
        Integer(10, 200, name='maxIter2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':50, 'epochs_pretrain':1001, 'epochs':251, 'space':
       [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-5, 10**-1, "log-uniform", name='lr'),
        Real(10**-1, 10**3, "log-uniform", name='alpha1'),
        Integer(10, 200, name='maxIter1'),
        Real(10**-1, 10**3, "log-uniform", name='alpha2'),
        Integer(10, 200, name='maxIter2')]},
  # 5
  {'model':run_ssc, 'dataset':'yaleB', 'n_rand':10, 'space':
       [Real(10**-1, 10**3, "log-uniform", name='alpha'),
        Integer(10, 200, name='maxIter')]},
  {'model':run_ssc, 'dataset':'Coil20', 'n_rand':10, 'space':
       [Real(10**-1, 10**3, "log-uniform", name='alpha'),
        Integer(10, 200, name='maxIter')]},
  {'model':run_ae, 'dataset':'yaleB', 'n_rand':10, 'epochs_pretrain':201, 'epochs':101, 'space':
       [Real(10**-2, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-3, 10**-1, "log-uniform", name='lr'),
        Real(10**0, 10**2, "log-uniform", name='alpha2'),
        Integer(10, 32, name='maxIter2')]},
  {'model':run_ae, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':1001, 'epochs':251, 'space':
       [Real(10**-4, 10**0, "log-uniform", name='lr_pretrain'),
        Real(10**-5, 10**-1, "log-uniform", name='lr'),
        Real(10**-1, 10**3, "log-uniform", name='alpha2'),
        Integer(10, 200, name='maxIter2')]},
  {'model':run_ssc, 'dataset':'Coil20', 'n_rand':160, 'maxIter':63, 'space':
       [Real(10**0, 10**4, "log-uniform", name='alpha')]},
  # 10
  {'model':run_ae, 'dataset':'Coil20', 'n_rand':160, 'epochs_pretrain':1000, 'epochs':250,
   'maxIter2':63, 'space':
       [Real(10**-6, 10**-2, "log-uniform", name='lr_pretrain'),
        Real(10**-6, 10**-2, "log-uniform", name='lr'),
        Real(10**-3, 10**1, "log-uniform", name='lambda2'),
        Real(10**0, 10**4, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':160, 'maxIter1':63, 'epochs_pretrain':1000,
   'epochs':250, 'maxIter2':63, 'trainC':False, 'giveC':False, 'space':
       [Real(10**-6, 10**-2, "log-uniform", name='lr_pretrain'),
        Real(10**0, 10**4, "log-uniform", name='alpha1'),
        Real(10**-6, 10**-2, "log-uniform", name='lr'),
        Real(10**-2, 10**2, "log-uniform", name='lambda1'),
        Real(10**-3, 10**1, "log-uniform", name='lambda2'),
        Real(10**0, 10**4, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':160, 'maxIter1':63, 'epochs_pretrain':1000,
   'epochs':250, 'maxIter2':63, 'trainC':True, 'giveC':False, 'space':
       [Real(10**-6, 10**-2, "log-uniform", name='lr_pretrain'),
        Real(10**0, 10**4, "log-uniform", name='alpha1'),
        Real(10**-6, 10**-2, "log-uniform", name='lr'),
        Real(10**-2, 10**2, "log-uniform", name='lambda1'),
        Real(10**-3, 10**1, "log-uniform", name='lambda2'),
        Real(10**1, 10**5, "log-uniform", name='lambda3'),
        Real(10**0, 10**4, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':160, 'maxIter1':63, 'epochs_pretrain':1000,
   'epochs':250, 'maxIter2':63, 'trainC':True, 'giveC':True, 'space':
       [Real(10**-6, 10**-2, "log-uniform", name='lr_pretrain'),
        Real(10**0, 10**4, "log-uniform", name='alpha1'),
        Real(10**-6, 10**-2, "log-uniform", name='lr'),
        Real(10**-2, 10**2, "log-uniform", name='lambda1'),
        Real(10**-3, 10**1, "log-uniform", name='lambda2'),
        Real(10**1, 10**5, "log-uniform", name='lambda3'),
        Real(10**0, 10**4, "log-uniform", name='alpha2')]},
  {'model':run_ssc, 'dataset':'Coil20', 'n_rand':40, 'maxIter':250, 'space':
       [Real(0.138167211, 345.4180278, "log-uniform", name='alpha')]},
  # 15
  {'model':run_ae, 'dataset':'Coil20', 'n_rand':40, 'epochs_pretrain':4000, 'epochs':1000,
   'maxIter2':250, 'space':
       [Real(1.27E-04, 0.318601418, "log-uniform", name='lr_pretrain'),
        Real(2.33E-07, 5.82E-04, "log-uniform", name='lr'),
        Real(1.02E-04, 0.254411693, "log-uniform", name='lambda2'),
        Real(0.081703137, 204.2578428, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':40, 'maxIter1':250, 'epochs_pretrain':4000,
   'epochs':1000, 'maxIter2':250, 'trainC':False, 'giveC':False, 'space':
       [Real(1.43E-04, 0.357575522, "log-uniform", name='lr_pretrain'),
        Real(4.59E+00, 11485.88822, "log-uniform", name='alpha1'),
        Real(2.60E-05, 6.51E-02, "log-uniform", name='lr'),
        Real(4.93E-03, 1.23E+01, "log-uniform", name='lambda1'),
        Real(2.08E-03, 5.194360803, "log-uniform", name='lambda2'),
        Real(0.0739943, 184.9857491, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':40, 'maxIter1':250, 'epochs_pretrain':4000,
   'epochs':1000, 'maxIter2':250, 'trainC':True, 'giveC':False, 'space':
       [Real(3.84E-05, 0.095948566, "log-uniform", name='lr_pretrain'),
        Real(3.47E-02, 86.73747683, "log-uniform", name='alpha1'),
        Real(1.33E-04, 3.33E-01, "log-uniform", name='lr'),
        Real(9.65E-03, 2.41E+01, "log-uniform", name='lambda1'),
        Real(1.01E-04, 0.251475935, "log-uniform", name='lambda2'),
        Real(5.73E+01, 143174.8108, "log-uniform", name='lambda3'),
        Real(0.914379246, 2285.948116, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':40, 'maxIter1':250, 'epochs_pretrain':4000,
   'epochs':1000, 'maxIter2':250, 'trainC':True, 'giveC':True, 'space':
       [Real(2.47E-05, 0.06180687, "log-uniform", name='lr_pretrain'),
        Real(3.10E-01, 775.4878807, "log-uniform", name='alpha1'),
        Real(1.68E-05, 4.20E-02 , "log-uniform", name='lr'),
        Real(2.79E-04, 6.97E-01, "log-uniform", name='lambda1'),
        Real(1.58E-01, 395.3416309, "log-uniform", name='lambda2'),
        Real(2.57E+01, 64339.80507, "log-uniform", name='lambda3'),
        Real(5.705480547, 14263.70137, "log-uniform", name='alpha2')]},
  {'model':run_ssc, 'dataset':'Coil20', 'n_rand':10, 'maxIter':1000, 'space':
       [Real(0.19212143, 120.0758935, "log-uniform", name='alpha')]},
  # 20
  {'model':run_ae, 'dataset':'Coil20', 'n_rand':10, 'epochs_pretrain':16000, 'epochs':1000,
   'maxIter2':1000, 'space':
       [Real(3.59E-04, 0.224458864, "log-uniform", name='lr_pretrain'),
        Real(1.43E-06, 8.91E-04, "log-uniform", name='lr'),
        Real(2.31E-03, 1.446133437, "log-uniform", name='lambda2'),
        Real(0.116079808, 72.54988004, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'maxIter1':1000, 'epochs_pretrain':16000,
   'epochs':4000, 'maxIter2':1000, 'trainC':False, 'giveC':False, 'space':
       [Real(6.43E-04, 0.401940318, "log-uniform", name='lr_pretrain'),
        Real(3.76E-01, 235.1025287, "log-uniform", name='alpha1'),
        Real(6.59E-06, 4.12E-03, "log-uniform", name='lr'),
        Real(4.31E-03, 2.69E+00, "log-uniform", name='lambda1'),
        Real(3.25E-04, 0.20315806, "log-uniform", name='lambda2'),
        Real(0.118788777, 74.24298589, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'maxIter1':1000, 'epochs_pretrain':16000,
   'epochs':4000, 'maxIter2':1000, 'trainC':True, 'giveC':False, 'space':
       [Real(2.44E-05, 0.0152341, "log-uniform", name='lr_pretrain'),
        Real(5.11E+00, 3196.653351, "log-uniform", name='alpha1'),
        Real(9.12E-05, 5.70E-02, "log-uniform", name='lr'),
        Real(1.14E-02, 7.10E+00, "log-uniform", name='lambda1'),
        Real(3.77E-03, 2.356855392, "log-uniform", name='lambda2'),
        Real(2.66E+01, 16631.88335, "log-uniform", name='lambda3'),
        Real(0.901278051, 563.298782, "log-uniform", name='alpha2')]},
  {'model':run_model, 'dataset':'Coil20', 'n_rand':10, 'maxIter1':1000, 'epochs_pretrain':16000,
   'epochs':4000, 'maxIter2':1000, 'trainC':True, 'giveC':True, 'space':
       [Real(3.14E-04, 0.196086635, "log-uniform", name='lr_pretrain'),
        Real(1.30E+00, 813.5497687, "log-uniform", name='alpha1'),
        Real(2.61E-05, 1.63E-02, "log-uniform", name='lr'),
        Real(4.24E-03, 2.65E+00, "log-uniform", name='lambda1'),
        Real(6.87E-05, 0.042940063, "log-uniform", name='lambda2'),
        Real(1.53E+03, 958405.1012, "log-uniform", name='lambda3'),
        Real(4.385245516, 2740.778447, "log-uniform", name='alpha2')]}
]