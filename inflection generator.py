#!/usr/bin/env python3
# coding: utf-8
# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

from inflection_generator import main, get_argparser


if __name__ == "__main__":
    args = get_argparser().parse_args('')
    args.kind = 'DPS'
    main(args)
