import os
from ConfigParser import SafeConfigParser

def configParse(file_path) :

    ### File check
    if not os.path.exists(file_path) :
        raise IOError(file_path)

    parser = SafeConfigParser()
    parser.read(file_path)

    ### Convert to dictionary
    config = {}
    for sect in parser.sections() :
        config[sect] = {}
        for opt in parser.options(sect) :
            config[sect][opt] = parser.get(sect, opt)

    ### Data check and convert from type
    for sect in config_settings.keys() :

#        if not sect in config :
#            raise KeyError(sect)

        for opt_attr in config_settings[sect] :
            if opt_attr['required'] and (not opt_attr['name'] in config[sect]) :
                raise KeyError(opt_attr['name'])

            if config[sect][opt_attr['name']] == 'None' :
                config[sect][opt_attr['name']] = None
            else :
                config[sect][opt_attr['name']] = opt_attr['type'](config[sect][opt_attr['name']])

    return config

