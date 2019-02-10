from itertools import combinations
import re
from pkg_resources import resource_stream, resource_filename

from caf import collect
from caf.app import UnfinishedTask
from caf.Tools import geomlib
from caf.Tools.geomlib import Atom, Molecule
from vdwsets import get_s22, get_x23, get_s66x8, get_s12l, Dataset, Cluster

from .app import app, aims
from .utils import chunks
from .functasks import get_results, get_grid, get_grid2, integrate_atomic_vv

np = None

aims_binary = 'aims-c68bd2f'
aims_binary_atoms = 'aims-138e95e'
aims_v3 = 'aims-e7c3eb6'
aims_v4 = 'aims-661b2f1'
aims_v5 = 'aims-c9fe4b2'
aims_master = 'aims-fbf4c4a'
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
    import pandas as pd
    ds.load_geoms()
    coros = []
    for key, cluster in ds.clusters.items():
        for fragment, geom in cluster.fragments.items():
            coros.append(taskgen(ds.name.lower(), key, fragment, geom))
    data = await collect(*coros)
    data = [x for x in data if x is not None]
    data = pd \
        .DataFrame(data, columns='label scale fragment data gridfile'.split()) \
        .set_index('label scale fragment'.split())
    return data, ds


@app.route('x23')
async def get_x23_set():
    data, ds = await get_dataset(get_x23())
    Cs = [0.0093, 0.0101]
    alpha = await collect(*(
        integrate_atomic_vv(dsname='x23', C=C, label=f'x23/vv/{C}')
        for C in Cs
    ))
    return data, ds, dict(zip(Cs, alpha))


@app.route('s66')
async def get_s66_set():
    data, ds = await get_dataset(get_s66x8())
    Cs = [0.0093, 0.0101]
    alpha = await collect(*(
        integrate_atomic_vv(dsname='s66', C=C, label=f's66/vv/{C}')
        for C in Cs
    ))
    return data, ds, dict(zip(Cs, alpha))


@app.route('s12l')
async def get_s12l_set():
    data, ds = await get_dataset(get_s12l())
    Cs = [0.0093, 0.0101]
    alpha = await collect(*(
        integrate_atomic_vv(dsname='s12l', C=C, label=f's12l/vv/{C}')
        for C in Cs
    ))
    return data, ds, dict(zip(Cs, alpha))


@app.route('C6s')
async def get_C6s_set():
    import pandas as pd
    all_coords = pd.read_hdf(resource_filename(__name__, 'data/C6-data.h5'), 'coords')
    coros = []
    for system, coords in all_coords.groupby('system'):
        geom = Molecule.from_coords(
            coords.species, coords[['x', 'y', 'z']].values
        )
        coros.append(taskgen('C6s', (system, 1.), 'main', geom))
    data = await collect(*coros)
    data = [x for x in data if x is not None]
    data = pd \
        .DataFrame(data, columns='label scale fragment data gridfile'.split()) \
        .set_index('label scale fragment'.split())
    return data


async def taskgen(dsname, key, fragment, geom):
    key_label = '_'.join(map(lambda k: str(k).lower().replace(' ', '-'), key))
    label = f'{dsname}/{key_label}/{fragment}'
    tags = default_tags.copy()
    if hasattr(geom, 'lattice'):
        tags['k_grid'] = geom.get_kgrid(0.8)
    aims_version = aims_binary
    if key == ('Benzene ... AcOH', 1.0) and fragment == 'fragment-1':
        aims_version = aims_v4
    tags['charge'] = geom.flags.get('charge', 0.)
    dft_task = await aims.task(
        geom=geom,
        basis='tight',
        aims=aims_version,
        tier=2,
        label=label,
        tags=tags,
        extra_feat=[join_grids],
    )
    my_get_grid = get_grid2 if dsname == 's12l' else get_grid
    results, grid_task = await collect(
        get_results(dft_task['results.xml'], label=f'{label}/results'),
        my_get_grid(dft_task['grid.xml'], label=f'{label}/grid')
    )
    return (
        *key, fragment, results,
        str(grid_task['grid.h5'].path) if grid_task else None
    )


async def taskgen_solids(geom, tags, root, data, ds, label, scale, atoms):
    dft_task = await aims.task(
        geom=geom,
        aims=aims_binary,
        basis='tight',
        tier=2,
        tags=tags,
        label=root,
        extra_feat=[join_grids],
    )
    results, grid_task = await collect(
        get_results(dft_task['results.xml'], label=f'{root}/results'),
        get_grid(dft_task['grid.xml'], label=f'{root}/grid')
    )
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


@app.route('solids')
async def get_solids():
    global np
    import numpy as np
    import pandas as pd

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
    coros = []
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
            coros.append(taskgen_solids(geom, tags, root, data, ds, label, scale, atoms))
    await collect(*coros)
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
            try:
                dft_task = await aims.task(
                    geom=geom,
                    aims=aims_binary_atoms,
                    basis='tight',
                    tier=3,
                    tags=tags,
                    label=label,
                    extra_feat=[join_grids],
                )
            except UnfinishedTask:
                continue
            results, = await collect(
                get_results(dft_task['results.xml'], label=f'{label}/results')
            )
            data['atoms'].append((
                atom_row.Z, atom_row.name, atom_row.configuration, conf, results,
            ))
    data['atoms'] = pd \
        .DataFrame(data['atoms'], columns='Z symbol full_conf conf data'.split()) \
        .set_index('symbol')
    return data, ds


async def taskgen_layered(geom, tags, root, data, label, shift):
    dft_task = await aims.task(
        geom=geom,
        aims=aims_master,
        basis='tight',
        tags=tags,
        label=root,
    )
    results, = await collect(
        get_results(dft_task['results.xml'], label=f'{root}/results'),
    )
    data.append((label, shift, results))


@app.route('layered')
async def get_layered():
    global np
    import numpy as np
    shifts = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 5, 10, 40]
    tags = {**default_tags, **solids_tags}
    del tags['total_energy_method']
    del tags['output']
    del tags['override_illconditioning']
    tags['many_body_dispersion_nl'] = {'beta': 0.81}
    coros = []
    data = []
    for label in ['MoS2', 'MoSe2', 'WS2', 'WSe2']:
        geom_base = geomlib.readfile(
            resource_filename(__name__, f'data/layered_geoms/{label.lower()}.vasp'),
        )
        tags['k_grid'] = (40, 40, 6)
        for shift in shifts:
            root = f'layered/{label}/{shift}'
            geom = geom_base.copy()
            xyz = geom.xyz
            xyz[:, 2] = xyz[:, 2] + np.where(xyz[:, 2] < geom.lattice[2][2]/2, 0, shift/2)
            geom.xyz = xyz
            geom.lattice[2] = (0, 0, geom.lattice[2][2]+shift)
            coros.append(taskgen_layered(geom, tags, root, data, label, shift))
    await collect(*coros)
    return data


@app.route('surface')
async def get_surface():
    global np
    import numpy as np
    tags = {
        **default_tags,
        **solids_tags,
        'many_body_dispersion_dev': {
            'grid_atoms': (1, 54, 31, 28, 27, 4),
            'grid_out': True
        },
        'many_body_dispersion_rsscs': {'beta': 0.81},
    }
    del tags['total_energy_method']
    del tags['output']
    del tags['override_illconditioning']
    tags['occupation_type'] = ('gaussian', 0.05)
    tags['charge_mix_param'] = 0.05
    tags['k_grid'] = (8, 8, 1)
    tags['sc_iter_limit'] = 100
    ds = get_s22()
    bz = geomlib.readfile(
        ds.clusters['Benzene dimer T-shaped', 1].fragments['fragment-1']
    ).centered().rotated('y', 90)
    slab = (
        get_slab(4.069, 'Ag')
        .supercell((1, 1, 2))
        .shifted((0, 0, -0.1))
        .normalized()
    )
    slab.lattice[2] = (0, 0, 40)
    height = slab.xyz[:, 2].max()
    hcp_site = slab.lattice[0][0]*2/9*5

    async def task(dist):
        geom = slab + bz.shifted((hcp_site, 0, height+dist))
        root = f'surface/{dist}'
        dft_task = await aims.task(
            geom=geom,
            aims=aims_master,
            basis='tight',
            tags=tags,
            label=root,
            extra_feat=[join_grids],
        )
        return dist, await collect(
            get_results(dft_task['results.xml'], label=f'{root}/results'),
            get_grid(dft_task['grid.xml'], label=f'{root}/grid')
        )

    return dict(await collect(*map(task, (2.7, 3, 3.15, 3.3, 3.6, 4, 5, 6, 7, 8, 10, 14))))


def get_slab(lattice, species):
    geom = get_crystal('A1', lattice, [species])
    lattice = np.array(geom.lattice)
    lattice = [
        3*lattice[0]-3*lattice[1],
        3*lattice[0]-3*lattice[2],
        lattice[0]+lattice[1]+lattice[2],
    ]
    geom = geom.supercell((12, 12, 12))
    frac = geom.xyz@np.linalg.inv(lattice)
    geom = geomlib.Crystal([
        a for a, b in zip(
            geom,
            np.all(((0, 0, 3) < frac+1e-10) & (frac+1e-10 < (1, 1, 4)), 1)
        ) if b
    ], lattice).normalized()
    a, _, b = map(np.linalg.norm, geom.lattice)
    new_latt = np.array([
        (np.sqrt(3)/2*a, 1/2*a, 0),
        (np.sqrt(3)/2*a, -1/2*a, 0),
        (0, 0, b)
    ])
    rotmat = (np.linalg.inv(geom.lattice)@new_latt).T
    geom = geom.rotated(rotmat=rotmat)
    return geom


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
