import argparse


parser = argparse.ArgumentParser(description="Arguments for running the scripts")

# ARGS FOR LOADING THE DATASET

parser.add_argument(
    "--obs", type=int, default=5, help="number of model's input frames"
)

parser.add_argument(
    "--preds", type=int, default=10, help="number of model's output frames"
)

parser.add_argument(
    "--hidden_dim", type=list, default=[128, 64, 32], help="hidden layer dimensions"
)

parser.add_argument(
    "--entities",
    type=int,
    default=11,
    choices=[11, 6],
    help="number of players to use",
)

parser.add_argument(
    "--input_dim", type=int, default=2, help="number of dimensions of the input"
)

parser.add_argument(
    "--batch_size", type=int, default=128, help="number of dimensions of the input"
)

# ARGS FOR THE MODEL

parser.add_argument(
    "--n_stgcnn_layers", type=int, default=9, help="number of stgcnn layers"
)
parser.add_argument(
    "--n_ccnn_layers",
    type=int,
    default=2,
    help="number of layers for the Coordinate-Channel Convolution",
)
parser.add_argument(
    "--n_tcnn_layers",
    type=int,
    default=5,
    help="number of layers for the Time-Extrapolator Convolution",
)
parser.add_argument(
    "--ccnn_kernel_size", type=list, default=[1, 1], help=" kernel for the C-CNN layers"
)
parser.add_argument(
    "--tcnn_kernel_size",
    type=list,
    default=[3, 3],
    help=" kernel for the Time-Extrapolator CNN layers",
)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=40,
    help="dimensions for the coordinates of the embedding",
)

parser.add_argument(
    "--st_gcnn_dropout", type=float, default=0.1, help="st-gcnn dropout"
)
parser.add_argument("--ccnn_dropout", type=float, default=0.0, help="ccnn dropout")

parser.add_argument("--tcnn_dropout", type=float, default=0.0, help="tcnn dropout")

parser.add_argument(
    "--skip_rate",
    type=int,
    default=1,
    choices=[1, 5],
    help="rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW",
)

# ARGS FOR THE TRAINING

parser.add_argument(
    "--euclidean_distance",
    action="store_true",
    dest="euclidean_distance",
    help="If true train euclidean distance model",
)
parser.set_defaults(euclidean_distance=False)


parser.add_argument(
    "--order_nn",
    action="store_true",
    dest="order_nn",
    help="If true train order_nn model",
)
parser.set_defaults(order_nn=False)

parser.add_argument(
    "--ordering1",
    action="store_true",
    dest="ordering1",
    help="If true train order_nn model",
)
parser.set_defaults(ordering1=False)


parser.add_argument(
    "--ordering2",
    action="store_true",
    dest="ordering2",
    help="If true train order_nn model",
)
parser.set_defaults(ordering2=False)


parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "test", "viz"],
    help="Choose to train,test or visualize from the model.Either train,test or viz",
)

parser.add_argument(
    "--n_epochs", type=int, default=100, help="number of epochs to train"
)

parser.add_argument(
    "--batch_size_test", type=int, default=8, help="batch size for the test set"
)
parser.add_argument(
    "--lr", type=float, default=1e-02, help="Learning rate of the optimizer"
)
parser.add_argument(
    "--use_scheduler", type=bool, default=True, help="use MultiStepLR scheduler"
)
parser.add_argument(
    "--milestones",
    type=list,
    default=[15, 25, 35, 40],
    help="the epochs after which the learning rate is adjusted by gamma",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    help="gamma correction to the learning rate, after reaching the milestone epochs",
)
parser.add_argument(
    "--clip_grad", type=float, default=None, help="select max norm to clip gradients"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./checkpoints",
    help="directory with the models checkpoints ",
)

# FAGS FOR LOGGING
parser.add_argument(
    "--wandb",
    action="store_true",
    dest="wandb",
    help="If true log on wandb.",
)
parser.set_defaults(wandb=False)

# * Model hyperparameters.

parser.add_argument(
    "--reg",
    type=float,
    default=0.0001,
    help="Parameter for curriculum loss' penalization.",
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.5,
    help="Parameter for curriculum loss' penalization.",
)

parser.add_argument(
    "--scale",
    type=float,
    default=0.001,
    help="Parameter for curriculum loss' penalization.",
)

parser.add_argument(
    "--model",
    type=str,
    choices=["euclidean", "order_nn", "e2e", "frozen"],
    help="Select which model you want to train.",
)


parser.add_argument(
    "--save",
    action="store_true",
    dest="save",
    help="If true save the model.",
)
parser.set_defaults(save=False)


parser.add_argument(
    "--gt",
    action="store_true",
    dest="gt",
    help="If true ground truth is used.",
)
parser.set_defaults(gt=False)

parser.add_argument(
    "--initialize",
    action="store_true",
    dest="initialize",
    help="If true initialize the model.",
)
parser.set_defaults(initialize=False)


parser.add_argument(
    "--supervised",
    action="store_true",
    dest="supervised",
    help="If true supervised the model.",
)
parser.set_defaults(supervised=False)




args = parser.parse_args()
    