from tqdm import tqdm
import torch
import os
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics  import precision_recall_curve, precision_score, f1_score, recall_score, accuracy_score
import torch.nn.functional as F
import numpy as np
def val(test_loader, model):
    tgt = []
    pred = []
    with tqdm(total = len(test_loader)) as tqdmer:
        for batch in test_loader:
            batch_pred, _, _, _, _ = model(batch)
            #
            tgt = tgt + batch['score'].tolist() 
            pred = pred + batch_pred.tolist() 
                #
            tqdmer.update(1) 
    #print('length: ', len(tgt))
    prd = np.clip(np.round(pred), 1, 6).astype(int).tolist()
    QWK = cohen_kappa_score(tgt, prd, weights = 'quadratic')
    F = f1_score(tgt, prd, average = 'micro')
    #P = precision_score(tgt, prd, average = 'micro')
    #R = recall_score(tgt, prd, average = 'micro')
    #ACC = accuracy_score(tgt, prd)
    #return QWK, P, R, F, ACC
    return QWK, F
    
def trainer(args,
            train_loader,
            val_loader,
            test_loader,
            model,
            optimizer,
            scheduler
            ):
    best_dev_QWK = 0.0
    for epoch in range(args.n_epochs):
        # set train mode
        print('epoch: ', epoch)
        model.train()
        iter_sample = 0.0
        L = 0.0
        L2 = 0.0
        L3 = 0.0
        L4 = 0.0
        with tqdm(total = len(train_loader)) as tqdmer:
            tqdmer.set_description('train_epoch:{}/{}'.format(epoch, args.n_epochs))
            for batch in train_loader:
                iter_sample += 1
                _, loss_mse, gdc, loss_cl, loss_aux = model(batch)
                loss = loss_mse - 0.1 * gdc + 0.1 * loss_cl + 0.1*loss_aux
                model.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                L += loss_mse.item()
                L2 += 0.1 * gdc.item()
                L3 += 0.1 * loss_cl.item()
                L4 += 0.1 * loss_aux.item()
                tqdmer.set_postfix(L1 = '{:.8f}'.format(L / iter_sample), \
                    L2 = '{:.8f}'.format(L2 / iter_sample), 
                    L3 = '{:.8f}'.format(L3 / iter_sample), 
                    L4 = '{:.8f}'.format(L4 / iter_sample))                  
                tqdmer.update(1) # 设置你每一次想让进度条更新的iteration 大小
            
        model.eval()
        QWK, F = val(val_loader, model)
        t_QWK, t_F = val(test_loader, model)
        print("QWK: ", QWK, "F: ", F)
        print("t_QWK: ", t_QWK, "t_F: ", t_F)
        #print('best_dev_QWK: ', best_dev_QWK)
        #print(float(QWK) - best_dev_QWK)
        if float(QWK) > float(best_dev_QWK):
            best_dev_QWK, best_dev_F = QWK, F
            best_test_QWK, best_test_F = t_QWK, t_F
            print('Hi~~~')
        print('best_dev_QWK:  ', best_dev_QWK,  '    best_dev_F:  ', best_dev_F)
        print('best_test_QWK: ', best_test_QWK, '    best_test_F: ', best_test_F)
        model.train() 