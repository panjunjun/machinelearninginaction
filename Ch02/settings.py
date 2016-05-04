DEBUG = True


def debug(variable_object):
    if DEBUG:
        #print "globals()=%s" % globals
        variable_name = None
        for k, v in globals().items():
            if v is variable_object:
                variable_name = k
                #print variable_name
        if variable_name:
            print "%s=%s" % (variable_name, variable_object)
        else:
            print variable_object


if __name__ == '__main__':
    c = {"a": 1, "b": 2}
    DEBUG = True
    debug(DEBUG)
    debug(c)
    DEBUG = False
    debug(DEBUG)

