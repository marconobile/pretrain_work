from typing import List


class MyArgPrsr(object):
    r'''
    Example usage in target file
    As global scope:
    parser_entries = [{'identifiers': ["-p", '--path'], 'type': str, 'help': 'The path to the file'}]

    As local scope:
    args = MyArgPrsr(parser_entries)
    path = args.path
    '''
    def __init__(self, parser_entries:List):
      from argparse import ArgumentParser
      parser = ArgumentParser()
      for el in parser_entries:
        if isinstance(el['identifiers'], list):
          for _identifier in el['identifiers']: assert isinstance(_identifier, str)
        elif isinstance(el['identifiers'], str): pass
        else: raise ValueError(f'identifiers not correct')
        assert isinstance(el['type'], type), f"type provided: {el['type']} is not a type"
        assert isinstance(el.get('help', ""), str), f"the help msg provided is not a str"
        if el.get('default', None): assert isinstance(el.get('default', None), el['type'])
        if el.get('optional', None): assert isinstance(el.get('optional'), bool)
        parser.add_argument(
            *el['identifiers'] if isinstance(el['identifiers'], list) else el['identifiers'], # if identifier are [-t, --tmp] then we can access it via self.args.tmp or self.args.t
            type=el['type'],           # data type used to interpret the inpt frm cmd line
            help=el.get('help', ''),
            default=el.get('default', None),
            nargs='?' if el.get('optional', False) else None
          )
      self.args = parser.parse_args()

    def __getattr__(self, arg: str): return getattr(self.args, arg)