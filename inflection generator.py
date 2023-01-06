#!/usr/bin/env python3
# coding: utf-8
# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

from inflection_generator import get_argparser, generate


if __name__ == "__main__":
    args = get_argparser().parse_args(['--kind', 'DPS'])
    generate(args)
