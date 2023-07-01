    
import torch
from tqdm import tqdm
from sklearn import metrics

def evaluate(model,dataset_loader):
    
    model.eval()

    prob_list= []
    rating_list = []
    eval_loss_list = []


    loss_func = torch.nn.BCELoss()

    pbar = tqdm(total = len(dataset_loader),desc = "",position=0, leave=True)

    for inputs in dataset_loader:
        with torch.no_grad():
            probs = model(
                user_indices=inputs['user_embed_id'],
                item_indices=inputs['movie_embed_id']
                )
            
            rating = inputs['rating'].view(-1,1)
            loss = loss_func(probs, rating)
            eval_loss_list.append(loss.item())

            probs = probs.cpu().numpy().flatten().tolist()
            prob_list.extend(probs)

            rating = rating.numpy().flatten().tolist()
            rating_list.extend(rating)

            pbar.update(1)

    pbar.close()

    eval_metrics = {}
    eval_metrics['eval_loss']= sum(eval_loss_list)/len(eval_loss_list)
    eval_metrics['eval_mse'] = metrics.mean_squared_error(rating_list,prob_list)
    
    return eval_metrics
    
    