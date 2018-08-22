from .app import dir_python


@dir_python.function_task
def get_results(output):
    from mbdvv.aimsparse import parse_xml

    return parse_xml(output)


@dir_python.function_task
def get_grid(gridfile):
    from mbdvv.aimsparse import parse_xml
    import pandas as pd

    data = parse_xml(gridfile)['point']
    keys = [k for k in data[0] if k not in {'dweightdr', 'dweightdh'}]
    df = pd.DataFrame([[pt[k][0] for k in keys] for pt in data], columns=keys)
    df.to_hdf('grid.h5', 'grid')


@dir_python.function_task
def get_grid2(gridfile):
    from mbdvv.aimsparse import parse_xmlelem
    import pandas as pd
    import xml.etree.ElementTree as ET

    it = (elem for _, elem in ET.iterparse(gridfile) if elem.tag == 'point')
    cols = {
        k: [v[0]] for k, v in parse_xmlelem(next(it)).items()
        if k not in {'dweightdr', 'dweightdh'}
    }
    for elem in it:
        parsed = parse_xmlelem(elem)
        for k, col in cols.items():
            col.append(parsed[k][0])
        for child in elem.iter():
            child.clear()
        elem.clear()
    df = pd.DataFrame(cols)
    df.to_hdf('grid.h5', 'grid')


@dir_python.function_task
def integrate_atomic_vv(dsname=None):
    from pymbd import MBDCalc
    import pandas as pd
    from mbdvv.app import app
    from mbdvv.physics import terf, calc_vvpol

    with app.context():
        df = app.get(dsname)[0]

    def rgrad_cutoff(rgrad, alpha):
        return 1-(1-terf(rgrad, k=60, x0=0.12))*terf(alpha-10*rgrad, k=6, x0=0.7)
    with MBDCalc(4) as mbd_calc:
        freq = mbd_calc.omega_grid[0]

    def f(x):
        return (
            pd
            .concat(
                dict(x.apply(lambda x: pd.read_hdf(x) if x else None)),
                names='label scale fragment i_point'.split()
            )
            .set_index('i_atom', append=True)
            .pipe(calc_vvpol, freq, rgrad_cutoff)
            .groupby('scale fragment i_atom'.split()).sum()
        )
    df.gridfile.groupby('label').apply(f).to_hdf('alpha.h5', 'alpha')
