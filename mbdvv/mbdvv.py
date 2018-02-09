import numpy as np
import pandas as pd
from itertools import islice, combinations
import re
from pkg_resources import resource_stream

from caflib.Configure import function_task
from caflib.Tools import geomlib
from caflib.Tools.geomlib import Atom
from caflib.Tools.aims import AimsTask
from caflib.Caf import Caf
from vdwsets import get_s22, get_s66x8, Dataset, Cluster

from .aimsparse import parse_xml

kcal = 627.503
ev = 27.2113845

default_tags = dict(
    xc='pbe',
    spin='none',
    relativistic=('zora', 'scalar', 1e-12),
    charge=0.,
    k_offset=(0.5, 0.5, 0.5),
    occupation_type=('gaussian', 0.02),
    mixer='pulay',
    n_max_pulay=10,
    charge_mix_param=0.6,
    sc_accuracy_eev=1e-3,
    sc_accuracy_rho=1e-5,
    sc_accuracy_etot=1e-6,
    sc_iter_limit=50,
    empty_states=10,
)
atom_tags = dict(
    spin='collinear',
    default_initial_moment='hund',
    occupation_type=('gaussian', 1e-6),
    occupation_acc=1e-10,
)
solids_tags = dict(
    output='hirshfeld_new',
    xml_file='results.xml',
    override_illconditioning=True,
)

app = Caf()
app.paths = [
    'solids/<>/*/*',
    'solids/<>/*/*/<>',
    's22/*/*',
    's22/*/*/<>',
    's66/*/*',
    's66/*/*/<>',
]
aims = AimsTask()


def get_dataset(ds, ctx):
    ds.load_geoms()
    data = []
    for key, cluster in ds.clusters.items():
        for fragment, geom in cluster.fragments.items():
            label = f'{"_".join(map(lambda k: str(k).lower().replace(" ", "-"), key))}/{fragment}'
            dft_task = ctx(
                features=[aims],
                geom=geom,
                basis='tight',
                aims='aims.master',
                label=label,
                tags={
                    **default_tags,
                    'output': 'hirshfeld_new',
                    'xml_file': 'results.xml',
                }
            )
            results_task = ctx(
                command='cp aims.xml results.xml',
                inputs=[('aims.xml', dft_task.outputs['results.xml'])],
                label=f'{label}/results',
            )
            data.append((
                *key,
                fragment,
                parse_xml(results_task.outputs['results.xml'])
                if results_task.finished else None,
            ))
    data = pd \
        .DataFrame(data, columns='label scale fragment data'.split()) \
        .set_index('label scale fragment'.split())
    return data, ds


@app.register('s22')
def get_s22_set(ctx):
    return get_dataset(get_s22(), ctx)


@app.register('s66')
def get_s66_set(ctx):
    return get_dataset(get_s66x8(), ctx)


@app.register('solids')
def get_solids(ctx):
    df_solids = pd.read_csv(resource_stream(__name__, 'data/solids.csv'), index_col='label scale'.split())
    df_atoms = pd.read_csv(resource_stream(__name__, 'data/atoms.csv'), index_col='symbol')
    ds = Dataset('solids', df_solids)
    all_atoms = set()
    data = {'solids': [], 'atoms': []}
    for solid in df_solids.itertuples():
        label = solid.Index[0]
        if label == 'Th':
            continue
        atoms = [
            ''.join(c) for c in
            chunks(re.split(r'([A-Z])', label)[1:], 2)
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
            root = f'crystals/{label}/{scale}'
            dft_task = ctx(
                features=[aims],
                geom=geom,
                aims='aims.master',
                basis='tight',
                tags=tags,
                label=root,
            )
            results_task = ctx(
                command='cp aims.xml results.xml',
                inputs=[('aims.xml', dft_task.outputs['results.xml'])],
                label=f'{root}/results',
            )
            data['solids'].append((
                label,
                scale,
                'crystal',
                parse_xml(results_task.outputs['results.xml'])
                if results_task.finished else None,
            ))
            for atom in atoms:
                data['solids'].append((label, scale, atom, None))
            ds[label, scale] = Cluster(
                intene=lambda x, species=geom.species: (
                    x['crystal']-sum(x[sp] for sp in species)
                )/len(species)
            )
    data['solids'] = pd \
        .DataFrame(data['solids'], columns='label scale fragment data'.split()) \
        .set_index('label scale fragment'.split())
    for species in all_atoms:
        atom = df_atoms.loc[species]
        conf = atom.configuration
        while conf[0] == '[':
            conf = df_atoms.loc[conf[1:3]].configuration + '/' + conf[4:]
        geom = geomlib.Molecule([Atom(atom.name, (0, 0, 0))])
        for conf, force_occ_tag in get_force_occs(conf.split('/')).items():
            label = f'atoms/{atom.name}/{conf}'
            tags = {
                **default_tags,
                **atom_tags,
                'force_occupation_basis': force_occ_tag
            }
            dft = ctx(
                features=[aims],
                geom=geom,
                aims='aims.master',
                basis='tight',
                tags=tags,
                label=label,
            )
            calc = get_energy(
                dft.outputs['run.out'], label=f'{label}/energy', ctx=ctx
            )
            data['atoms'].append((
                atom.Z,
                atom.name,
                atom.configuration,
                conf,
                calc.result/ev if calc.finished else None
            ))
    data['atoms'] = pd \
        .DataFrame(data['atoms'], columns='Z symbol full_conf conf ene'.split()) \
        .set_index('symbol')
    return data, ds


@app.register('atoms')
def get_atoms(ctx):
    df_atoms = pd.read_csv(resource_stream(__name__, 'data/atoms.csv'), index_col='symbol')
    table = []
    for atom in df_atoms.itertuples():
        if atom.Index in ['Ce']:
            continue
        conf = atom.configuration
        while conf[0] == '[':
            conf = df_atoms.loc[conf[1:3]].configuration + '/' + conf[4:]
        geom = geomlib.Molecule([Atom(atom.Index, (0, 0, 0))])
        for conf, force_occ_tag in get_force_occs(conf.split('/')).items():
            label = f'{atom.Index}/{conf}'
            tags = {
                **default_tags,
                **atom_tags,
                'force_occupation_basis': force_occ_tag,
                'xc': 'pbe0',
                'output': 'hirshfeld_new',
                'xml_file': 'results.xml',
                'k_offset': None,
            }
            dft_task = ctx(
                features=[aims],
                geom=geom,
                aims='aims.dd96b0c',
                basis='tight',
                tags=tags,
                label=label,
            )
            results_task = ctx(
                command='cp aims.xml results.xml',
                inputs=[('aims.xml', dft_task.outputs['results.xml'])],
                label=f'{label}/results',
            )
            table.append((
                atom.Z,
                conf,
                atom.Index,
                parse_xml(results_task.outputs['results.xml']) if results_task.finished else None,
            ))
    table = pd \
        .DataFrame(table, columns='Z conf symbol results'.split()) \
        .set_index('Z conf'.split())
    return table


@function_task
def get_energy(output):
    with open(output) as f:
        next(l_ for l_ in f if l_ == '  Self-consistency cycle converged.\n')
        return float(next(l for l in f if 'Total energy uncorrected' in l).split()[5])


def get_force_occs(conf):
    orb = 'spdf'
    shells = [
        (int(shell[0]), orb.index(shell[1]), int(shell[2:] or 1))
        for shell in conf
    ]
    nmaxocc = sum(2*l+1 for n, l, occ in shells)
    spin_shells = []
    for n, l, occ in shells:
        for spin in (1, 2):
            spin_occ = min(occ, 2*l+1)
            occ -= spin_occ
            spin_shells.append((n, l, spin, spin_occ))
            if occ == 0:
                break
    force_occs = []
    unfilled = []
    for i_shell, (n, l, spin, occ) in enumerate(spin_shells):
        if occ < 2*l+1:
            unfilled.append(i_shell)
            continue
        for m in range(-l, l+1):
            force_occs.append((1, spin, 'atomic', n, l, m, 1, nmaxocc))
    assert len(unfilled) <= 1
    if len(unfilled) == 0:
        return {'-': force_occs}
    n, l, spin, occ = spin_shells[unfilled[0]]
    all_force_occs = {}
    for ms in combinations(range(-l, l+1), occ):
        all_force_occs[','.join(map(str, ms))] = force_occs + [
            (1, spin, 'atomic', n, l, m, 1, nmaxocc) for m in ms
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


def chunks(iterable, n):
    iterable = iter(iterable)
    while True:
        chunk = list(islice(iterable, n))
        if not chunk:
            break
        yield chunk
