import numpy as np
import yaml
import pickle

studies = ['helmi', 'pace', 'massari', 'kallivayalil', 'simon']
names = []
study_from = []
MC_set = []

for study in studies:
    dwarf_file = 'data/dwarfs/'+study+'.yaml'
    with open(dwarf_file, 'r') as f:
        dwarfs = yaml.load(f)
    study_names = list(dwarfs.keys())
    MC_dwarfs = np.load('data/sampling/'+study+'_converted.npy')

    for name in study_names:
        if name not in names:
            names.append(name)
            study_from.append(study)
            MC_set.append(MC_dwarfs[study_names.index(name)])
MC_set = np.array(MC_set)

fritz_file = 'data/dwarfs/fritz.yaml'
with open(fritz_file, 'r') as f:
    fritz = yaml.load(f)
fritz = list(fritz.keys())

assert np.isin(names, fritz).all()
MC_dwarfs = np.load('data/sampling/fritz_converted.npy')
MC_set_fritz = []
for name in names:
    MC_set_fritz.append(MC_dwarfs[fritz.index(name)])

np.save('data/sampling/HPMKS_converted', MC_set)
np.save('data/sampling/fritz_HPMKS_converted', MC_set_fritz)

map = {'name': names, 'study': study_from}
with open('data/sampling/HPMKS_key.pkl', 'wb') as f:
    pickle.dump(map, f, protocol=pickle.HIGHEST_PROTOCOL)

"""
# for printing Table 1
dwarf_file = 'data/dwarfs/dwarf_props.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)

for name in dwarfs:
    dwarf = dwarfs[name]
    print(name+' & '+str(dwarf['ra'])+' & '+str(dwarf['dec'])+' & '+\
            str(dwarf['abs_mag'])+' & $'+str(dwarf['distance'])+' \pm '+\
            str(dwarf['distance_error'])+'$ & $'+str(dwarf['vel_los'])+' \pm '+\
            str(dwarf['vel_los_error'])+'$ \\\\')
"""
