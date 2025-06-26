from podgen.mcmc_utils import generate

if __name__ == "__main__":

    from pathlib import Path

    from scripts.eval_utils import load_model

    batch_size = 128
    model_path = "/public/home/wangqingchang/PODGen/output/hydra/singlerun/2025-06-24/mp_20"

    model, _, _ = load_model(model_path, load_data=True)
    