OK_FORMAT = True

test = {   'name': 'p1',
    'points': 2,
    'suites': [   {   'cases': [   {   'code': '>>> _dw, _db = gradiente_logistica(X_t, es_cero, w0, b0)\n'
                                               '>>> _r = torch.sigmoid(X_t @ w0 + b0) - es_cero\n'
                                               ">>> assert tuple(_dw.shape) == (64,), 'dw tiene la forma de w (n = 64)'\n"
                                               ">>> assert torch.allclose(_dw, X_t.T @ _r / len(es_cero)), 'dw debe coincidir con Xᵀ(σ(Xw + b) − y)/M'\n"
                                               ">>> assert abs(float(_db) - _r.mean().item()) < 1e-10, 'db debe coincidir con el promedio de σ(z) − y'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
