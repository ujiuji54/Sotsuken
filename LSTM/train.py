from argparse import ArgumentParser
import yaml
import numpy as np
from fx_replicator import (
    load_wave, load_aux, flow, build_model, LossFunc, train
)

def main():

    args = parse_args()

    with open(args.config_file) as fp:
        config = yaml.safe_load(fp)
    
    input_timesteps = int(config["sampling_rate"]*config["input_time"]/1000)
    output_timesteps = int(config["sampling_rate"]*config["input_time"]/1000)
    batch_size = config["batch_size"]
    max_epochs = config["max_epochs"]
    patience = config["patience"]

    train_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1), load_aux(_[2]))
        for _ in config["train_data"]]
    train_dataflow = flow(train_dataset, input_timesteps, batch_size)
    print(train_dataset[0])

    val_dataset = [
        (load_wave(_[0]).reshape(-1, 1), load_wave(_[1]).reshape(-1, 1), load_aux(_[2]))
        for _ in config["val_data"]]
    val_dataflow = flow(val_dataset, input_timesteps, batch_size)
   
    model = build_model(input_timesteps)
    model.compile(
        loss=LossFunc(output_timesteps),
        optimizer="adam")

    train(model, train_dataflow, val_dataflow, max_epochs, patience)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_file", "-c", default="./config.yml",
        help="configuration file (*.yml)")
    return parser.parse_args()

if __name__ == '__main__':
    main()
