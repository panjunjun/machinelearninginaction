DEBUG = False


def dbug(variable_object):
    if DEBUG:
        for k, v in globals().iteritems():
            print "variable_name=%s, variable_value=%s" % (k, v)
        print "---------------------spliting---------------------"
        for k, v in locals().iteritems():
            print "variable_name=%s, variable_value=%s" % (k, v)
        variable_name = None
        for k, v in globals().items():
            if v is variable_object:
                variable_name = k
                # print variable_name
        if variable_name:
            print "%s=%s" % (variable_name, variable_object)
        else:
            print variable_object


def debug(object_to_print):
    if DEBUG:
        print object_to_print


if __name__ == '__main__':
    c = {"a": 1, "b": 2}
    DEBUG = True
    debug(DEBUG)
    debug(c)
    DEBUG = False
    debug(DEBUG)

