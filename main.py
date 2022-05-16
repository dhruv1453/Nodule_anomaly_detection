from datasets import *
from trainers import *
import numpy as np
from torch.utils.data import DataLoader
import random
import argparse
from torchvision import transforms
import mlflow
# import pytorch_lightning as pl
torch.autograd.set_detect_anomaly(True)


def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.cuda.set_device(device)
    print('Device: '+str(device))

    exp = {"dir_datasets": args.dir_datasets,
           "dir_out": args.dir_out,
           "load_weigths": args.load_weigths, "epochs": args.epochs, "item": args.item, "method": args.method,
           "input_shape": args.input_shape,
           "batch_size": 8, "lr": 1e-05, "zdim": 32, "images_on_ram": True, "wkl": 1, "wr": 1, "wadv": 0, "wc": 0,
           "wae": args.wae, "epochs_to_test": 50, "dense": True, "expansion_loss": True, "log_barrier": True,
           "channel_first": True, "normalization_cam": args.normalization_cam, "avg_grads": True, "t": 20, "context": False,
           "level_cams": -4, "p_activation_cam": args.p_activation_cam, "bayesian": False, "loss_reconstruction": "L2",
           "hist_match": True,
           "expansion_loss_penalty": args.expansion_loss_penalty, "restoration": False, "threshold_on_normal": False, "n_blocks": 5}

    metrics = []
    for iteration in [0]:

        TRANSFORM_IMG = transforms.Compose([
            transforms.Resize(exp['input_shape'][-1]),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.38169734018417545, 0.35629716336767236, 0.3216164951658236],
            #                     std=[0.23080161182849712, 0.22390099134794886, 0.20951167832871959]),
            # transforms.Grayscale()
        ])

        # Loading Data
        train_data = torchvision.datasets.ImageFolder(root=exp['dir_datasets'], transform=TRANSFORM_IMG)
        train_data_loader = DataLoader(train_data, batch_size=exp['batch_size'], shuffle=True, num_workers=4)

        # Set trainer and train model
        trainer = WSALTrainer(exp["dir_out"], item=exp["item"], method=exp["method"], zdim=exp["zdim"], lr=exp["lr"],
                              input_shape=exp["input_shape"], expansion_loss=exp["expansion_loss"], wkl=exp["wkl"],
                              wr=exp["wr"], wadv=exp["wadv"], wae=exp["wae"], epochs_to_test=exp["epochs_to_test"],
                              load_weigths=exp["load_weigths"],
                              n_blocks=exp["n_blocks"], dense=exp["dense"], log_barrier=exp["log_barrier"],
                              normalization_cam=exp["normalization_cam"], avg_grads=exp["avg_grads"], t=exp["t"],
                              context=exp["context"], level_cams=exp["level_cams"], iteration=iteration,
                              p_activation_cam=exp["p_activation_cam"], bayesian=exp["bayesian"],
                              loss_reconstruction=exp["loss_reconstruction"],
                              expansion_loss_penalty=exp["expansion_loss_penalty"], restoration=exp["restoration"],
                              threshold_on_normal=exp["threshold_on_normal"])

        # Save experiment setup
        with open(exp['dir_out'] + 'setup.json', 'w') as fp:
            json.dump(exp, fp)
            
        # Auto log all MLflow entities
        # mlflow.pytorch.autolog()

        # Train
        # with mlflow.start_run() as run:
        trainer.train(train_data_loader, exp['epochs'])

        # Save overall metrics
        metrics.append(list(trainer.metrics.values()))

    # Compute average performance and save performance in dictionary

    metrics = np.array(metrics)
    metrics_mu = np.mean(metrics, 0)
    metrics_std = np.std(metrics, 0)

    labels = list(trainer.metrics.keys())
    metrics_mu = {labels[i]: metrics_mu[i] for i in range(0, len(labels))}
    metrics_std = {labels[i]: metrics_std[i] for i in range(0, len(labels))}

    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_avg_val.json', 'w') as fp:
        json.dump(metrics_mu, fp)
    with open(exp['dir_out'] + exp['item'][0] + '/' + 'metrics_std_val.json', 'w') as fp:
        json.dump(metrics_std, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_datasets", default='nodule_anomaly_split/negative/split_data', type=str)
    parser.add_argument("--dir_out", default="nodule_anomaly_split/results/ae/color30", type=str)
    parser.add_argument("--method", default="ae", type=str)    # ae, vae, anoVAEGAN, f_ano_gan, proposed
    parser.add_argument("--expansion_loss_penalty", default="log_barrier", type=str)
    parser.add_argument("--normalization_cam", default="sigm", type=str)
    parser.add_argument("--item", default=[""], type=list)
    parser.add_argument("--load_weigths", default=False, type=bool)
    parser.add_argument("--epochs", default=30, type=int)
    parser.add_argument("--t", default=20, type=int)
    parser.add_argument("--wae", default=5, type=int)
    parser.add_argument("--p_activation_cam", default=0.2, type=float)
    parser.add_argument("--input_shape", default=[3, 128, 128], type=list)

    args = parser.parse_args()
    main(args)
