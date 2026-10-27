OK_FORMAT = True

test = {   'name': 'p3',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': ">>> assert abs(mase(np.array([5.0, 7.0]), np.array([4.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0])) - 1.2) < 1e-12, 'MAE = (1 + 3)/2 = 2; escala = (2 + 1 "
                                               "+ 2)/3 = 5/3; MASE = 1.2'\n"
                                               ">>> assert abs(mase(np.array([5.0, 7.0]), np.array([4.0, 4.0]), np.array([1.0, 3.0, 2.0, 4.0]), m=2) - 2.0) < 1e-12, 'con m = 2 la escala es (|2 − 1| "
                                               "+ |4 − 3|)/2 = 1'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
