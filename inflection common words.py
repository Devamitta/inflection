#!/usr/bin/env python3
# coding: utf-8 

# FIXME It is common to avoid whitespaces in names of modules, better to use underscore

from inflection_generator import main, _get_argparser


if __name__ == "__main__":
    args = _get_argparser().parse_args('')
    args.kind = 'SBS'
    main(args)
