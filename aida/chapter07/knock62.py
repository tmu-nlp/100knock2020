from pprint import pprint

from knock60 import load_model, obtain_vector

if __name__ == '__main__':
    model = load_model()
    src_word = 'United_States'
    pprint(model.most_similar(src_word))

"""
[('Unites_States', 0.7877248525619507),
 ('Untied_States', 0.7541370391845703),
 ('United_Sates', 0.74007248878479),
 ('U.S.', 0.7310774326324463),
 ('theUnited_States', 0.6404393911361694),
 ('America', 0.6178410053253174),
 ('UnitedStates', 0.6167312264442444),
 ('Europe', 0.6132988929748535),
 ('countries', 0.6044804453849792),
 ('Canada', 0.6019070148468018)]
"""

