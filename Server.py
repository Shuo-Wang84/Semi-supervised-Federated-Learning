import copy
import torch

def FedAvg(w, client_freq):
    with torch.no_grad():
        w_avg = copy.deepcopy(w[0])
        #sup_p = client_freq[0]/sum(client_freq)
        num_takepart = len(w)
        ratio_takepart = sum(client_freq[:num_takepart])
        ratio = [freq / ratio_takepart for freq in client_freq[:num_takepart]]

        for net_id in range(num_takepart):
            if net_id == 0:
                for key in w[net_id]:
                    w_avg[key] = w[net_id][key] * ratio[net_id]
            else:
                for key in w[net_id]:
                    w_avg[key] += w[net_id][key] * ratio[net_id]
        return w_avg

