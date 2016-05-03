DEBUG = True


def debug(string):
    if DEBUG:
        print string


if __name__ == '__main__':
    DEBUG = True
    debug('debug = True')
    DEBUG = False
    debug('debug = False')