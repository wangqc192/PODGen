# Convert CIFs from the PODGen to CSV file, and can check the repeat structures.
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure, Lattice
from p_tqdm import p_umap, p_map
from functools import partial
import os
import glob



def process_cif_text(cif_file):
    return Structure.from_str(cif_file, fmt='cif').get_primitive_structure()

def process_poscar(poscar):
    return Structure.from_str(poscar, fmt='poscar').get_primitive_structure()

def process_cif(cif_file):
    return Structure.from_file(cif_file).get_primitive_structure()
    
matcher = StructureMatcher()

#++++++++++++++++++++++++++ input ++++++++++++++++++++++++++
# If a database already exists in this field, you can convert it into a CSV file to be used for preliminary duplication checks of newly generated crystal structures.
file_path = '' 
in_path = '/data/home/cyye/5-materialgen/7-cdvae/CrystalFormer_torch/scripts/genout/topo_TI'  # the output path of PODGen
out_file = '/data/work/cyye/0-project/11-PODGen/PODGen/testfile/test.csv'  # the output file
hadgen_path = '' # The path to previously generated crystal structures, used for duplication checking. If not set, duplication checking will be skipped.
worker = 5 # set the number of workers according to your actual needs.


#++++++++++++++++++++++++++ prepare ++++++++++++++++++++++++++
exist_csv = []
for root, _, files in os.walk(hadgen_path): 
    for file in files:
        if file.endswith('.csv'):  
            exist_csv.append(os.path.join(root, file))  
if len(exist_csv) > 0:
    print('Use this data for duplication checking:')
    print(exist_csv)

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    cif = list(df['cif'])

    ref_structs = p_map(process_cif_text, cif, num_cpus=worker)
    ref_alphabetical_formula = list(df['alphabetical_formula'])


cif_files = glob.glob(os.path.join(in_path, '*.cif'))
cif_files = list(cif_files)
in_structs = p_map(process_cif, cif_files, num_cpus=worker)

had_cif = []
if os.path.exists(out_file):
    df = pd.read_csv(out_file)
    had_cif = list(df['cif'])


#++++++++++++++++++++++++++ duplication checking ++++++++++++++++++++++++++

if os.path.exists(file_path):
    after_structs = []
    for in_struct in in_structs:
        alphabetical_formula = in_struct.composition.alphabetical_formula

        indices = [index for index, value in enumerate(ref_alphabetical_formula) if value == alphabetical_formula]
        # print(len(indices), flush=True)
        is_match = False
        for idx in indices:
            if matcher.fit(in_struct, ref_structs[idx]):
                is_match = True
                break
        
        if is_match:
            continue
        
        after_structs.append(in_struct)
else:
    after_structs = in_structs


for i in range(len(exist_csv)):
    print(f'start {i} in {len(exist_csv)}')
    df = pd.read_csv(exist_csv[i])
    poscar = list(df['poscar'])
    ref_alphabetical_formula = list(df['formula'])
    ref_structs = p_map(process_poscar, poscar, num_cpus=worker)
    in_structs = after_structs
    after_structs = []
    for in_struct in in_structs:
        alphabetical_formula = in_struct.composition.alphabetical_formula

        indices = [index for index, value in enumerate(ref_alphabetical_formula) if value == alphabetical_formula]
        # print(len(indices), flush=True)
        is_match = False
        for idx in indices:
            if matcher.fit(in_struct, ref_structs[idx]):
                is_match = True
                break
        
        if is_match:
            continue
        
        after_structs.append(in_struct)
    in_structs = after_structs


need_alphabetical_formula_list = [struc.composition.alphabetical_formula for struc in in_structs]
after_structs = []
for i in range(len(in_structs)):
    in_struct = in_structs[i]
    alphabetical_formula = need_alphabetical_formula_list[i]

    indices = [index for index, value in enumerate(need_alphabetical_formula_list[:i]) if value == alphabetical_formula]
    is_match = False
    for idx in indices:
        if matcher.fit(in_struct, in_structs[idx]):
            is_match = True
            break
    
    if is_match:
        continue
    
    after_structs.append(in_struct)


# after_structs = [struc.to(fmt="cif") for struc in after_structs]
after_structs = p_map(lambda struc: struc.to(fmt="cif"), after_structs, num_cpus=worker)
had_cif.extend(after_structs)

print(f'had_cif {len(had_cif)}')

out_df = {'cif': had_cif}
out_df = pd.DataFrame(out_df)
out_df.to_csv(out_file, index=False)

