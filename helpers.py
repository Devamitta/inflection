import string


def excel_index(index: int) -> str:
    alphabet = list(string.ascii_uppercase)
    alph_len = len(alphabet)
    alphabet.insert(0, '0')

    result = ''
    correction = 1
    div = -1

    while div:
        div = index // (alph_len)
        rem = index % (alph_len) + correction
        index = div

        result = alphabet[rem] + result

        if len(result) == 1:
            correction = 0

    return result
