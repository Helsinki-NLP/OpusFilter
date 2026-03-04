"""Command-line interface for opusfilter-diagram command."""
import argparse
import collections
import json
import logging
import os
import sys

from graphviz import Digraph

from opusfilter.util import yaml, yaml_dumps, get_inputs, get_outputs, get_other_params


logger = logging.getLogger(__name__)


def main(args=None):
    """Main entry point for opusfilter-diagram command."""
    parser = argparse.ArgumentParser(prog='opusfilter-diagram',
        description='Draw a diagram from OpusFilter configuration')

    parser.add_argument('yaml', metavar='FILE', help='YAML configuration file')
    parser.add_argument('output', metavar='FILE', help='output file (rendered if does not end with .dot)')
    parser.add_argument('--rankdir', default='LR', choices=['TB', 'LR'], help='graph direction (default %(default)s)')
    parser.add_argument('--exclude-params', action='store_true', help='do not write step parameters')

    args = parser.parse_args(args)

    logging.basicConfig(level=logging.INFO)

    config = yaml.load(open(args.yaml, 'r', encoding='utf-8'))

    graph = Digraph(comment=os.path.basename(args.yaml))
    graph.attr(rankdir=args.rankdir)
    graph.attr('node', shape='box')

    node_outputs = {}
    used_outputs = set()
    sources = {}
    targets = {}
    for idx, step in enumerate(config['steps']):
        name = str(idx)
        label = step['type']
        params = get_other_params(step)
        if params and not args.exclude_params:
            paramstr = yaml_dumps(params)
            label += '\n\n' + paramstr.replace('\n', '\l')
        graph.node(name, label=label)
        for fname in get_inputs(step):
            if fname in node_outputs:
                graph.edge(node_outputs[fname], name, label=os.path.basename(fname))
                used_outputs.add(fname)
            elif fname in sources:
                graph.edge(sources[fname], name)
            else:
                source_name = 'source_{}'.format(len(sources))
                sources[fname] = source_name
                #graph.node(source_name, label=os.path.dirname(fname), shape='plaintext')
                graph.node(source_name, shape='point')
                graph.edge(source_name, name, label=os.path.basename(fname))
        for fname in get_outputs(step):
            node_outputs[fname] = name
    for fname in node_outputs:
        if fname not in used_outputs:
            target_name = 'target_{}'.format(len(targets))
            targets[fname] = target_name
            graph.node(target_name, shape='point')
            graph.edge(node_outputs[fname], target_name, label=os.path.basename(fname))

    if args.output.endswith('.dot'):
        with open(args.output, 'w', encoding='utf-8') as fobj:
            fobj.write(graph.source)
        logger.info("Wrote %s", args.output)
    else:
        base, ext = os.path.splitext(args.output)
        out = graph.render(filename=base, format=ext.lstrip('.'), cleanup=True, view=False)
        logger.info("Wrote %s", out)

    return 0


if __name__ == '__main__':
    sys.exit(main())
