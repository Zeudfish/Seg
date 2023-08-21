import torch
from torch import nn
from tqdm import tqdm
import sys
import time
from segmentation_models_pytorch.utils.meter import AverageValueMeter

def train_one_epoch(model,data_loader,device,epoch,optimizer,scheduler,losses,metrics):
    model.train()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        image,target = data
        image,target=image.to(device),target.to(device)
        predict=model(image)
        loss=losses(predict,target)
        
        loss.backward()
        
        
        optimizer.step()

        optimizer.zero_grad()

        scheduler.step()


        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,  loss.item() / (step + 1))


    return loss
    

def evaluate(model,data_loader,device,epoch,optimizer,scheduler,losses,metrics):
    model.eval()
    data_loader = tqdm(data_loader, file=sys.stdout)
    sample_num = 0
    inference_time = 0.0
    for metric in metrics:
            metric.to(device)
    
    logs = {}
    loss_meter = AverageValueMeter()
    avg_inference_time_meter = AverageValueMeter()
    metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
    for step, data in enumerate(data_loader):
        image,target = data
        image,target=image.to(device),target.to(device)
        sample_num += image.shape[0]

        start_time = time.time()
        predict=model(image)
        end_time = time.time()
        inference_time += (end_time - start_time) 
        loss=losses(predict,target)
        loss_value = loss.cpu().detach().numpy()

        loss_meter.add(loss_value)
        loss_logs = {losses.__name__: loss_meter.mean}
        logs.update(loss_logs)
        for metric_fn in metrics:
            metric_value = metric_fn(predict, target).cpu().detach().numpy()
            metrics_meters[metric_fn.__name__].add(metric_value)
        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        logs.update(metrics_logs)

        avg_inference_time = inference_time / sample_num
        avg_inference_time_meter.add(avg_inference_time)

        avg_inference_time_logs={"avg_inference_time":avg_inference_time_meter.mean}
        logs.update(avg_inference_time_logs)

        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        data_loader.set_postfix_str(s)



    return logs
