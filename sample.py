from podgen.mcmc_utils import generate

if __name__ == "__main__":

    from pathlib import Path
    
    import pandas as pd
    import numpy as np

    from scripts.eval_utils import load_model

    batch_size = 128
    model_path = "/home/wangqc/PODGen/output/hydra/singlerun/2025-06-24/mp20"

    model, _, _ = load_model(Path(model_path), load_data=True)
    
    data = generate(batch_size, model, 225)
    data['lattice'].reshape(batch_size,-1)
    data['lattice']=data['lattice'].reshape(batch_size,-1)
    for i in data.keys():
        data[i] = data[i].tolist()
    new_data = {}
    for i in range(len(data['lattice'])):
        data['lattice'][i][-3:] = [j * 180/np.pi for j in data['lattice'][i][-3:]]
    new_data["L"] = data["lattice"]
    new_data["X"] = data["frac_coor"]
    new_data["A"] = data["atom_type"]
    new_data["W"] = data["wyckoff"]
    new_data["M"] = data["M"]
    df = pd.DataFrame(new_data)
 
    df.to_csv("/home/wangqc/PODGen/output_225.csv")
    
    