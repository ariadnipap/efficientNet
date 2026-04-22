import torch
import logging
import numpy as np
import random
import tqdm
import os
import copy

from src import dataset, models, utils, train_eval, params

args = params.get_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

weight_dir, plot_dir, pred_dir = utils.setup_logging(args, mode="train")
use_cuda = torch.cuda.is_available()

train_dataloader, val_dataloader, test_dataloader = dataset.get_dataloaders(args)
num_classes = train_dataloader.num_classes
print("Number of classes:", num_classes)

# Load model from checkpoint
model = models.get_model(args.model, output_dim=num_classes, pretrained=False)
weights = torch.load(args.model_weight)
model.load_state_dict(weights)
model.train()
if use_cuda:
    model = model.cuda()

geo_model = None

# Optimizer — use smaller lr since we're continuing training
if args.optim == "nesterov":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
elif args.optim == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

# Cosine decay from args.lr down to 10% of it
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.1*args.lr)

logging.info(str(args))

epoch_pbar = tqdm.tqdm(total=args.epochs, desc="epoch")
LOG_FMT = "Epoch {:3d} | {} set | LR {:.4E} | Loss {:.4f} | Acc {:.2f}"
best_val_metrics = None
best_weights = None
best_epoch = None
val_acc = -1

for epoch in range(args.epochs):
    cur_lr = scheduler.get_last_lr()[-1]

    train_loss, train_acc = train_eval.run_loop(
        args, train_dataloader, model,
        mode="train", optimizer=optimizer,
        use_cuda=use_cuda, epoch=epoch,
        geo_model=geo_model,
    )
    logging.info(LOG_FMT.format(epoch, "train", cur_lr, train_loss, 100*train_acc))

    if (epoch+1) % args.eval_freq == 0:
        utils.save_model(model, os.path.join(weight_dir, "checkpoint_{}.pt".format(epoch)))

        val_loss, val_acc, val_metrics = train_eval.run_loop(
            args, val_dataloader, model,
            mode="eval", use_cuda=use_cuda,
            geo_model=geo_model,
        )
        logging.info(LOG_FMT.format(epoch, "val", cur_lr, val_loss, 100*val_acc))
        logging.info(val_metrics)

        if best_val_metrics is None or utils.metric_str2acc(best_val_metrics) < utils.metric_str2acc(val_metrics):
            best_val_metrics = val_metrics
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    scheduler.step()
    epoch_pbar.update(1)
    epoch_pbar.set_description("LR: {:.4E} Loss: {:.4f} Acc: {:.4f} ValAcc: {:.4f}".format(cur_lr, train_loss, train_acc, val_acc))

logging.info("Best Val: Epoch {}, Best metrics".format(best_epoch))
logging.info(best_val_metrics)

# Test with best weights
model.load_state_dict(best_weights)
test_loss, test_acc, test_metrics = train_eval.run_loop(
    args, test_dataloader, model,
    mode="eval", use_cuda=use_cuda,
    geo_model=geo_model,
)
logging.info(LOG_FMT.format(best_epoch, "test", cur_lr, test_loss, 100*test_acc))
logging.info(test_metrics)
print("Test results:")
print(test_metrics)