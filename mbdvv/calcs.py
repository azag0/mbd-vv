import numpy as np
import pandas as pd
from itertools import combinations
import re
from pkg_resources import resource_stream

from caf.cellar import collect
from caf.Tools import geomlib
from caf.Tools.geomlib import Atom
from vdwsets import get_s22, get_s66x8, Dataset, Cluster

from .app import app, aims, dir_python
from .utils import chunks

app.paths.extend([
    'solids/<>/*/*',
    'solids/<>/*/*/<>',
    's22/*/*',
    's22/*/*/<>',
    's66/*/*',
    's66/*/*/<>',
])

aims_binary = 'aims-138e95e'
default_tags = dict(
    xc='pbe',
    spin='none',
    relativistic=('zora', 'scalar', 1e-12),
    total_energy_method='tpss',
    charge=0.,
    occupation_type=('gaussian', 0.02),
    mixer='pulay',
    n_max_pulay=10,
    charge_mix_param=0.6,
    sc_accuracy_eev=1e-3,
    sc_accuracy_rho=1e-5,
    sc_accuracy_etot=1e-6,
    sc_iter_limit=50,
    empty_states=10,
    output=('hirshfeld_new', '.true.'),
    xml_file='results.xml',
)
atom_tags = dict(
    spin='collinear',
    default_initial_moment='hund',
    occupation_type=('gaussian', 1e-6),
    occupation_acc=1e-10,
)
solids_tags = dict(
    override_illconditioning=True,
    k_offset=(0.5, 0.5, 0.5),
)


def join_grids(task):
    task['command'] += (
        ' && head -2 grid-0.xml >grid.xml && for f in grid-*.xml; '
        'do tail -n +3 -q $f | head -n -1 >>grid.xml; '
        'done && tail -1 grid-0.xml >>grid.xml && rm grid-*.xml'
    )


async def get_dataset(ds):
    ds.load_geoms()
    coros = []
    for key, cluster in ds.clusters.items():
        for fragment, geom in cluster.fragments.items():
            coros.append(taskgen(ds.name.lower(), key, fragment, geom))
    data = await collect(coros)
    data = pd \
        .DataFrame(data, columns='label scale fragment data gridfile'.split()) \
        .set_index('label scale fragment'.split())
    return data, ds


@app.route('s22')
async def get_s22_set():
    return await get_dataset(get_s22())


@app.route('s66')
async def get_s66_set():
    return await get_dataset(get_s66x8())


async def taskgen(dsname, key, fragment, geom):
    key_label = '_'.join(map(lambda k: str(k).lower().replace(' ', '-'), key))
    label = f'{dsname}/{key_label}/{fragment}'
    dft_task = await aims.task(
        geom=geom,
        basis='tight',
        aims=aims_binary,
        tier=2,
        label=label,
        tags=default_tags,
        extra_feat=[join_grids],
    )
    results, grid_task = await collect([
        get_results(dft_task['results.xml'], label=f'{label}/results'),
        get_grid(dft_task['grid.xml'], label=f'{label}/grid')
    ])
    return (
        *key, fragment, results,
        str(grid_task['grid.h5'].path) if grid_task else None
    )


@app.route('solids')
async def get_solids():
    df_solids = pd.read_csv(
        resource_stream(__name__, 'data/solids.csv'),
        index_col='label scale'.split()
    )
    df_atoms = pd.read_csv(
        resource_stream(__name__, 'data/atoms.csv'),
        index_col='symbol'
    )
    ds = Dataset('solids', df_solids)
    all_atoms = set()
    data = {'solids': [], 'atoms': []}
    for solid in df_solids.itertuples():
        label = solid.Index[0]
        if label == 'Th':
            continue
        atoms = [
            ''.join(c) for c in chunks(re.split(r'([A-Z])', label)[1:], 2)
        ]
        all_atoms.update(atoms)
        tags = {**default_tags, **solids_tags}
        n_kpts = solid.k_grid
        tags['k_grid'] = (n_kpts, n_kpts, n_kpts)
        if not np.isnan(solid.default_initial_moment):
            tags['spin'] = 'collinear'
            tags['default_initial_moment'] = solid.default_initial_moment
        if not np.isnan(solid.occupation_type):
            tags['occupation_type'] = ('gaussian', solid.occupation_type)
        if not np.isnan(solid.charge_mix_param):
            tags['charge_mix_param'] = solid.charge_mix_param
        for scale in np.linspace(0.97, 1.03, 7):
            lattice = solid.lattice_pbe \
                if not np.isnan(solid.lattice_pbe) \
                else solid.lattice
            lattice *= scale
            geom = get_crystal(solid.structure, lattice, atoms)
            root = f'solids/crystals/{label}/{scale}'
            dft_task = await aims.task(
                geom=geom,
                aims=aims_binary,
                basis='tight',
                tier=2,
                tags=tags,
                label=root,
                extra_feat=[join_grids],
            )
            results, grid_task = await collect([
                get_results(dft_task['results.xml'], label=f'{root}/results'),
                get_grid(dft_task['grid.xml'], label=f'{root}/grid')
            ])
            data['solids'].append((
                label, scale, 'crystal', results,
                str(grid_task['grid.h5'].path) if grid_task else None,
            ))
            for atom in atoms:
                data['solids'].append((label, scale, atom, None, None))
            ds[label, scale] = Cluster(
                intene=lambda x, species=geom.species: (
                    x['crystal']-sum(x[sp] for sp in species)
                )/len(species)
            )
    data['solids'] = pd \
        .DataFrame(data['solids'], columns='label scale fragment data gridfile'.split()) \
        .set_index('label scale fragment'.split())
    for species in all_atoms:
        atom_row = df_atoms.loc[species]
        conf = atom_row.configuration
        while conf[0] == '[':
            conf = df_atoms.loc[conf[1:3]].configuration + '/' + conf[4:]
        geom = geomlib.Molecule([Atom(atom_row.name, (0, 0, 0))])
        force_occ_tags = get_force_occs(conf.split('/'))
        for conf, force_occ_tag in force_occ_tags.items():
            label = f'solids/atoms/{atom_row.name}/{conf}'
            tags = {
                **default_tags,
                **atom_tags,
                'force_occupation_basis': force_occ_tag
            }
            dft_task = await aims.task(
                geom=geom,
                aims=aims_binary,
                basis='tight',
                tier=3,
                tags=tags,
                label=label,
                extra_feat=[join_grids],
            )
            results, = await collect([
                get_results(dft_task['results.xml'], label=f'{label}/results')
            ])
            data['atoms'].append((
                atom_row.Z, atom_row.name, atom_row.configuration, conf, results,
            ))
    data['atoms'] = pd \
        .DataFrame(data['atoms'], columns='Z symbol full_conf conf data'.split()) \
        .set_index('symbol')
    return data, ds


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


def get_force_occs(conf):
    orb = 'spdf'
    shells = [
        (int(shell[0]), orb.index(shell[1]), int(shell[2:] or 1))
        for shell in conf
    ]
    nmaxocc = sum(2*l+1 for n, l, occ in shells)
    # spin_shells = []
    # for n, l, occ in shells:
    #     for spin in (1, 2):
    #         spin_occ = min(occ, 2*l+1)
    #         occ -= spin_occ
    #         spin_shells.append((n, l, spin, spin_occ))
    #         if occ == 0:
    #             break
    force_occs = []
    unfilled = []
    for i_shell, (n, l, occ) in enumerate(shells):
        if occ <= 2*l+1:
            for m in range(-l, l+1):
                force_occs.append((1, 2, 'atomic', n, l, m, 0, nmaxocc))
        if occ % (2*l+1) != 0:
            unfilled.append(i_shell)
        # for m in range(-l, l+1):
        #     force_occs.append((1, spin, 'atomic', n, l, m, 1, nmaxocc))
    assert len(unfilled) <= 1
    if len(unfilled) == 0:
        return {'-': force_occs}
    n, l, occ = shells[unfilled[0]]
    all_force_occs = {}
    for ms in combinations(range(-l, l+1), 2*l+1-(occ % (2*l+1))):
        all_force_occs[','.join(map(str, ms))] = force_occs + [
            (1, 1, 'atomic', n, l, m, 0, nmaxocc) for m in ms
        ]
    return all_force_occs


def get_crystal(group, a, species):
    if group == 'A2':
        latt_vecs = [(-a/2, a/2, a/2), (a/2, -a/2, a/2), (a/2, a/2, -a/2)]
    else:
        latt_vecs = [(0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]
    if group in ('A1', 'A2'):
        atoms = [Atom(species[0], (0, 0, 0))]
    elif group == 'A4':
        atoms = [
            Atom(species[0], (0, 0, 0)),
            Atom(species[0], (a/4, a/4, a/4)),
        ]
    elif group == 'B1':
        atoms = [
            Atom(species[0], (0, 0, 0)),
            Atom(species[1], (a/2, a/2, a/2)),
        ]
    elif group == 'B3':
        atoms = [
            Atom(species[0], (0, 0, 0)),
            Atom(species[1], (a/4, a/4, a/4)),
        ]
    return geomlib.Crystal(atoms, latt_vecs)
