#!/usr/bin/env python
from jinja2 import Environment
from pathlib import Path
import yaml
import sys
import os

env = Environment(
    block_start_string='<+',
    block_end_string='+>',
    variable_start_string='<<',
    variable_end_string='>>',
    comment_start_string='<#',
    comment_end_string='#>',
    trim_blocks=True,
    autoescape=False,
)
template = env.from_string(sys.stdin.read())
print(template.render(**yaml.load(Path(sys.argv[1]).read_text())))
