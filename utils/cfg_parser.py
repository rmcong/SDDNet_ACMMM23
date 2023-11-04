from configparser import *
from collections import OrderedDict
import time
import ast

# see configparser souce code here
# https://github.com/python/cpython/blob/master/Lib/configparser.py

class WithTimeInterpolation(ExtendedInterpolation):
    def _interpolate_some(self, parser, option, accum, rest, section, map,
                          depth):
        rawval = parser.get(section, option, raw=True, fallback=rest)
        if depth > MAX_INTERPOLATION_DEPTH:
            raise InterpolationDepthError(option, section, rawval)
        while rest:
            p = rest.find("$")
            if p < 0:
                accum.append(rest)
                return
            if p > 0:
                accum.append(rest[:p])
                rest = rest[p:]
            # p is no longer used
            c = rest[1:2]
            if c == "$":
                accum.append("$")
                rest = rest[2:]
            elif c == "{":
                m = self._KEYCRE.match(rest)
                if m is None:
                    raise InterpolationSyntaxError(option, section,
                        "bad interpolation variable reference %r" % rest)
                path = m.group(1).split(':')
                rest = rest[m.end():]
                sect = section
                opt = option
                try:
                    if len(path) == 1:
                        opt = parser.optionxform(path[0])
                        v = map[opt]
                    elif len(path) == 2:
                        sect = path[0]
                        opt = parser.optionxform(path[1])
                        ###################the part I modified################
                        if sect == '_TIME':
                            if parser.has_section(sect):
                                raise ValueError("'{:s}' is kept for time interpolation, not allowed \
                                to be used as section title".format(sect) )
                            else:
                                v = time.strftime(path[1], time.localtime())
                        else:
                            opt = parser.optionxform(path[1])
                            v = parser.get(sect, opt, raw=True)
                        #######################################################
                    else:
                        raise InterpolationSyntaxError(
                            option, section,
                            "More than one ':' found: %r" % (rest,))
                except (KeyError, NoSectionError, NoOptionError):
                    raise InterpolationMissingOptionError(
                        option, section, rawval, ":".join(path)) from None
                if "$" in v:
                    self._interpolate_some(parser, opt, accum, v, sect,
                                           dict(parser.items(sect, raw=True)),
                                           depth + 1)
                else:
                    accum.append(v)
            else:
                raise InterpolationSyntaxError(
                    option, section,
                    "'$' must be followed by '$' or '{', "
                    "found: %r" % (rest,))



class CfgParser(object):
    def __init__(self, path): 
#         https://docs.python.org/3/library/configparser.html
        self.config = ConfigParser(interpolation=WithTimeInterpolation(),
                                   comment_prefixes=('#',';'),
                                   inline_comment_prefixes=(';', '#'))
        self.config.read(path)
    
    def parse(self):
        config_dict = OrderedDict()
        for section in self.config.sections():
            section_dict = OrderedDict()
            for option in self.config[section]:
                # https://stackoverflow.com/a/3513475
                # print(f'no conversion: {self.config[section][option]}, type {type(self.config[section][option])}\n')
                section_dict[option] = ast.literal_eval(self.config[section][option])
                # print(f'converted: {section_dict[option]}, type {type(section_dict[option])}\n')
            config_dict[section] = section_dict
        return config_dict

    def set(self, section, option, value):
        self.config.set(section, option, value)


    def save(self, path):
        with open(path, 'w') as configfile:
            self.config.write(configfile)
