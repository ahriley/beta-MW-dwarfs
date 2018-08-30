import numpy as np
import yaml
import pickle

studies = ['helmi', 'pace', 'massari', 'kallivayalil', 'simon']
names = []
study_from = []
MC_set = []
magclouds = []
for study in studies:
    # get names, samplings for each study
    dwarf_file = 'data/'+study+'.yaml'
    with open(dwarf_file, 'r') as f:
        dwarfs = yaml.load(f)
    study_names = list(dwarfs.keys())
    MC_dwarfs = np.load('data/sampling/'+study+'_galacto.npy')

    # if satellite hasn't been added, add to study
    for name in study_names:
        if name not in names:
            names.append(name)
            study_from.append(study)
            MC_set.append(MC_dwarfs[study_names.index(name)])

            if name == 'LMC' or name == 'SMC':
                magclouds.append(MC_dwarfs[study_names.index(name)])
MC_set = np.array(MC_set)

# get Fritz version of the gold satellites
fritz_file = 'data/fritz.yaml'
with open(fritz_file, 'r') as f:
    fritz = yaml.load(f)
fritz = list(fritz.keys())
MC_dwarfs = np.load('data/sampling/fritz_galacto.npy')
MC_set_fritz = []
fritz_gold_names = []
for name in names:
    try:
        MC_set_fritz.append(MC_dwarfs[fritz.index(name)])
        fritz_gold_names.append(name)
    except:
        # Magellanic Clouds not in Fritz
        continue

# handle the Magellanic Clouds
MC_set_fritz = np.concatenate((MC_set_fritz, np.array(magclouds)))
MC_fritzplusMCs = np.concatenate((MC_dwarfs, np.array(magclouds)))
assert MC_set_fritz.shape == MC_set.shape

# save all the data
np.save('data/sampling/fritzplusMCs', MC_fritzplusMCs)
np.save('data/sampling/gold', MC_set)
np.save('data/sampling/fritz_gold', MC_set_fritz)

# map study to the order of the names
samples = ['fritzplusMCs', 'gold', 'fritz_gold']
fritz.extend(['LMC', 'SMC'])
fritz_gold_names.extend(['LMC', 'SMC'])
map = dict(zip(samples, (fritz, names, fritz_gold_names)))
with open('data/sampling/names_key.pkl', 'wb') as f:
    pickle.dump(map, f, protocol=pickle.HIGHEST_PROTOCOL)

"""
# for printing dwarf props Table
dwarf_file = 'data/dwarfs/dwarf_props.yaml'
with open(dwarf_file, 'r') as f:
    dwarfs = yaml.load(f)

name_study = dict(zip(names, study_from))
study_number = dict(zip(studies, [1, 5, 4, 3, 2]))

count = len(study_number) + 1
tags = []
for name in dwarfs:
    dwarf = dwarfs[name]
    if dwarf['ra'] == 0 or dwarf['vel_los_error'] == 0:
        continue
    elif name=='Eridanus II' or name=='Grus II' or name=='Pegasus III':
        continue

    refnums = []
    for cit in dwarf['citation'].split(','):
        ii = 0
        for char in cit:
            if str.isdigit(char):
                break
            ii += 1
        tag = cit[ii:]
        if tag not in study_number.keys():
            study_number[tag] = count
            count += 1
        refnums.append(study_number[tag])

    refstring = ""
    for num in sorted(refnums):
        refstring += '['+str(num)+'] '

    if name not in names:
        print(name+' & '+"{0:.3f}".format(dwarf['ra'])+' & '+\
                "{0:.3f}".format(dwarf['dec'])+' & '+\
                str(dwarf['abs_mag'])+' & $'+str(dwarf['distance'])+' \pm '+\
                str(dwarf['distance_error'])+'$ & $'+str(dwarf['vel_los'])+\
                ' \pm '+str(dwarf['vel_los_error'])+'$ & -- & '+refstring+\
                '\\\\')
    else:
        print(name+' & '+"{0:.3f}".format(dwarf['ra'])+' & '+\
            "{0:.3f}".format(dwarf['dec'])+' & '+\
            str(dwarf['abs_mag'])+' & $'+str(dwarf['distance'])+' \pm '+\
            str(dwarf['distance_error'])+'$ & $'+str(dwarf['vel_los'])+\
            ' \pm '+str(dwarf['vel_los_error'])+'$ & ['+\
            str(study_number[name_study[name]])+'] & '+refstring+'\\\\')

number_study = {}
for study in study_number.keys():
    number_study[study_number[study]] = study

for num in sorted(number_study.keys()):
    print('['+str(num)+'] \citet{'+str(number_study[num])+'}; ', end='')

for num in sorted(number_study.keys()):
    print(number_study[num])
# """
