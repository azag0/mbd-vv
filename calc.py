import pandas as pd
import csv
from itertools import islice
import re

from caflib.Configure import function_task
from caflib.Tools import geomlib2 as geomlib
from caflib.Tools.geomlib2 import Atom
from caflib.Tools.aims import AimsTask

from caflib.Caf import Caf

ev = 27.2107

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

calc = Caf()


@calc.register
def run(ctx):
    data = {'enes': [], 'vols': []}
    aims = AimsTask()
    with open('data/solids.csv') as f:
        reader = csv.DictReader(f, delimiter=';')
        rows = list(reader)
    for row in rows:
        for k, v in row.items():
            try:
                row[k] = int(v)
            except ValueError:
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        label = row['label']
        species = [
            ''.join(c) for c in
            chunks(re.split(r'([A-Z])', label)[1:], 2)
        ]
        geom = get_crystal(row['group'], row['lattice'], species)
        tags = default_tags.copy()
        n_kpts = row['k_grid']
        tags['k_grid'] = (n_kpts, n_kpts, n_kpts)
        if row['default_initial_moment']:
            tags['spin'] = 'collinear'
            tags['default_initial_moment'] = row['default_initial_moment']
        if row['occupation_type']:
            tags['occupation_type'] = ('gaussian', row['occupation_type'])
        if row['charge_mix_param']:
            tags['charge_mix_param'] = row['charge_mix_param']
        tags['dry_run'] = ()
        dft = ctx(
            features=[aims],
            geom=geom,
            aims='aims.master',
            basis='tight',
            tags=tags,
            label=label,
        )
        calc = get_energy(dft.outputs['run.out'], label=f'{label}/energy', ctx=ctx)
        if calc.finished:
            data['enes'].append((label, calc.result/ev))
        calc = get_volumes(dft.outputs['run.out'], label=f'{label}/volumes', ctx=ctx)
        if calc.finished:
            for i, vol in enumerate(calc.result):
                data['vols'].append((label, i, vol))
    data['enes'] = pd.DataFrame(data['enes'], columns='label ene'.split())
    data['vols'] = pd.DataFrame(data['vols'], columns='label i vol'.split())
    return data


@function_task
def get_energy(output):
    with open(output) as f:
        next(l_ for l_ in f if l_ == '  Self-consistency cycle converged.\n')
        return float(next(l for l in f if 'Total energy uncorrected' in l).split()[5])


@function_task
def get_volumes(output):
    from math import nan

    ratios = []
    with open(output) as f:
        next(l for l in f if 'Performing Hirshfeld analysis' in l)
        for line in f:
            if not line.strip():
                break
            if 'Free atom volume' in line:
                free = float(line.split()[-1]) or nan
            elif 'Hirshfeld volume' in line:
                hirsh = float(line.split()[-1])
                ratios.append(hirsh/free)
    return ratios


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
