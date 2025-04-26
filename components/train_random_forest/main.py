import hydra
from omegaconf import DictConfig
from src.train_random_forest.run import go

@hydra.main(config_path=None, config_name="config")
def main(config: DictConfig):
    go(config)

if __name__ == "__main__":
    main()