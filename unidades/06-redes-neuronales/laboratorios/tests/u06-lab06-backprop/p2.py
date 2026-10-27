OK_FORMAT = True

test = {   'name': 'p2',
    'points': 1,
    'suites': [   {   'cases': [   {   'code': ">>> assert np.allclose(relu(np.array([[-2.0, 0.0, 3.0]])), [[0.0, 0.0, 3.0]]), 'ReLU(−2, 0, 3) = (0, 0, 3)'\n"
                                               ">>> assert np.allclose(relu_atras(np.array([5.0, 5.0, 5.0]), np.array([-2.0, 0.0, 3.0])), [0.0, 0.0, 5.0]), 'el gradiente pasa solo donde Z > 0 (en Z "
                                               "= 0 la derivada se toma como 0)'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
