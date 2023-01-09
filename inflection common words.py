#!/usr/bin/env python3
# coding: utf-8

# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

from inflection_generator.cli import get_argparser, generate_inflections


if __name__ == "__main__":
    args = get_argparser().parse_args(['--kind', 'SBS'])
    generate_inflections(args)
