from entrypoint import main, get_argparser

if __name__ == "__main__":
    ARGS = get_argparser().parse_args()
    main(ARGS)
