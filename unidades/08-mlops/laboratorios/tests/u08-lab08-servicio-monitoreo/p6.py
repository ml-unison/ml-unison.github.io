OK_FORMAT = True

test = {   'name': 'p6',
    'points': 3,
    'suites': [   {   'cases': [   {   'code': ">>> assert len(error_movil) == 12 and error_movil.index[0] == pd.Period('2012-01', freq='M'), 'un error por mes de 2012, empezando en enero'\n"
                                               ">>> _feb, _ene = (pd.Period('2011-02', freq='M'), pd.Period('2011-01', freq='M'))\n"
                                               ">>> _med = reentrenar_ventana(bicis, lambda: DummyRegressor(strategy='mean'), meses_ventana=1, primer_mes='2011-02')\n"
                                               ">>> _esp = mean_absolute_error(bicis.loc[bicis.periodo == _feb, 'total'], np.full((bicis.periodo == _feb).sum(), bicis.loc[bicis.periodo == _ene, "
                                               "'total'].mean()))\n"
                                               ">>> assert len(_med) == 23 and abs(_med.iloc[0] - _esp) < 1e-09, 'con ventana de 1 mes y un modelo que predice la media, el error de febrero de 2011 "
                                               "usa la media de enero'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}
