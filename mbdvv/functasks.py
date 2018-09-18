import pandas as pd

from .app import dir_python
from .physics import nm_cutoff, lg_cutoff, lg_cutoff2, reduced_grad, alpha_kin, vv_pol


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


def calc_vvpol(x, freq, **kwargs):
    idx = x.index
    n = x.rho.values
    grad = x.rho_grad_norm.values
    kin = x.kin_dens.values
    w = x.part_weight.values
    rgrad = reduced_grad(n, grad)
    alpha = alpha_kin(n, grad, kin)
    cutoff_nm = nm_cutoff(rgrad, alpha)
    cutoff_lg = lg_cutoff(rgrad, alpha)
    cutoff_lg2 = lg_cutoff2(rgrad, alpha)
    freq = freq[:, None]
    x = pd.concat({
        'vvpol': pd.DataFrame((vv_pol(n, grad, u=freq, **kwargs)*w).T),
        'vvpol_nm': pd.DataFrame((vv_pol(n, grad, u=freq, **kwargs)*(w*cutoff_nm)).T),
        'vvpol_lg': pd.DataFrame((vv_pol(n, grad, u=freq, **kwargs)*(w*cutoff_lg)).T),
        'vvpol_lg2': pd.DataFrame((vv_pol(n, grad, u=freq, **kwargs)*(w*cutoff_lg2)).T),
    }, axis=1)
    x.index = idx
    return x


def evaluate_vv_batch(key, x, freq, C):
    import pandas as pd

    from .functasks import calc_vvpol

    return key, (
        pd
        .concat(
            dict(x.apply(lambda x: pd.read_hdf(x) if x else None)),
            names='label scale fragment i_point'.split()
        )
        .assign(kin_dens=lambda x: x.kin_dens/2)
        .set_index('i_atom', append=True)
        .pipe(calc_vvpol, freq, C=C)
        .groupby('scale fragment i_atom'.split()).sum()
    )


@dir_python.function_task
def integrate_atomic_vv(dsname=None, C=None):
    from pymbd import MBDCalc
    import pandas as pd
    from mbdvv.app import app
    from mbdvv.functasks import evaluate_vv_batch
    from multiprocessing import Pool

    with app.context():
        df = app.get(dsname)[0]
    with MBDCalc(4) as mbd_calc:
        freq = mbd_calc.omega_grid[0]

    args_list = [(key, x, freq, C) for key, x in df.gridfile.groupby('label')]
    with Pool() as pool:
        pd.concat(
            dict(pool.starmap(evaluate_vv_batch, args_list))
        ).to_hdf('alpha.h5', 'alpha')
