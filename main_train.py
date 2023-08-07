import torch
from torch import nn
import faiss
import argparse
import pickle
import numpy as np
from torchsummary import summary
from sentence_transformers import SentenceTransformer
from model.densecap import densecap_resnet50_fpn
import random
from random import sample
from lib.data_loader import DenseCapDataset, DataLoaderPFG
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

writer = SummaryWriter(os.path.join("./results"))

MAX_EPOCHS = 200
random.seed(2023)
np.random.seed(2023)
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
MODEL_NAME = 'visual_text'

def caption_train(args):
    update_layer = ['roi_heads.box_describer.embedding_layer.weight', 
                    'roi_heads.box_describer.rnn.weight_ih_l0', 
                    'roi_heads.box_describer.rnn.weight_hh_l0', 
                    'roi_heads.box_describer.rnn.bias_ih_l0', 
                    'roi_heads.box_describer.rnn.bias_hh_l0', 
                    'roi_heads.box_describer.feature_project_layer.0.weight', 
                    'roi_heads.box_describer.feature_project_layer.0.bias', 
                    'roi_heads.box_describer.fc_layer.weight', 
                    'roi_heads.box_describer.fc_layer.bias']
    
    train_set = DenseCapDataset(args.img_root, args.data_path, args.lut_path, dataset_type='train')
    val_set = DenseCapDataset(args.img_root, args.data_path, args.lut_path, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']
    train_loader = DataLoaderPFG(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)
    val_loader = DataLoaderPFG(val_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)
    model = densecap_resnet50_fpn(backbone_pretrained=True,
                                  feat_size=384,
                                  hidden_size=4096,
                                  max_len=30,
                                  emb_size=384,
                                  rnn_num_layers=1,
                                  vocab_size=18516,
                                  fusion_type='init_inject',
                                  box_detections_per_img=50,
                                  init_training_step='C')
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    
    #visual encoder freeze
    for name, para in model.named_parameters():
        if name in update_layer:
            para.requires_grad = True
        else:
            para.requires_grad = False
            
    optimizer = torch.optim.Adam([{'params': (para for name, para in model.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in model.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args.caption_lr}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    
    iter_counter = 0
    best_map = 0.
    
    for epoch in range(MAX_EPOCHS):

        for batch, (img, targets, info) in enumerate(tqdm(train_loader)):

            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]

            model.train()
            losses = model(img, targets)
            caption_loss = losses['loss_caption']
            
            optimizer.zero_grad()
            caption_loss.backward()
            optimizer.step()
            
            if batch % 1000 == 0:
                print(f"batch:{batch}, caption_loss:{caption_loss}")
            iter_counter += 1
        save_model(model, optimizer, iter_counter, flag=str(epoch))

def em_train(args):
    train_set = DenseCapDataset(args.img_root, args.data_path, args.lut_path, dataset_type='train')
    val_set = DenseCapDataset(args.img_root, args.data_path, args.lut_path, dataset_type='val')
    idx_to_token = train_set.look_up_tables['idx_to_token']
    train_loader = DataLoaderPFG(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                 pin_memory=True, collate_fn=DenseCapDataset.collate_fn)
    args.num_cluster = args.num_cluster.split(',')
    print("sentence data loading...")
    with open(args.sentences_path, "rb") as f:
        sentences = pickle.load(f)
        
    print("EM-Step training...")
    text_encoder = SentenceTransformer(args.text_encoder, device=device)
    visual_encoder = densecap_resnet50_fpn(backbone_pretrained=True,
                                  feat_size=384,
                                  hidden_size=4096,
                                  max_len=30,
                                  emb_size=384,
                                  rnn_num_layers=1,
                                  vocab_size=18516,
                                  fusion_type='init_inject',
                                  box_detections_per_img=50,
                                  init_training_step='E')
    text_encoder.to(device)
    visual_encoder.to(device)
    optimizer = torch.optim.Adam([{'params': (para for name, para in visual_encoder.named_parameters()
                                    if para.requires_grad and 'box_describer' not in name)},
                                  {'params': (para for para in visual_encoder.roi_heads.box_describer.parameters()
                                              if para.requires_grad), 'lr': args.caption_lr}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    
    iter_counter = 0
    best_map = 0.
    
    for epoch in range(MAX_EPOCHS):
        #E-Step
        print(f"epoch: {epoch}")
        text_encoder.eval()
        cluster_result, _ = sentence_embedding(args, text_encoder, sentences, sen_emb_batch_size=1000, k_means_iter=40, n_cluster=1024)
        centroid = cluster_result['centroids']
        density = cluster_result['density']
        
        #M-step
        for batch, (img, targets, info) in enumerate(tqdm(train_loader)):
            img = [img_tensor.to(device) for img_tensor in img]
            targets = [{k:v.to(device) for k, v in target.items()} for target in targets]
            visual_encoder.train()
            losses = visual_encoder(img, targets)
            
            detect_loss =  losses['loss_objectness'] + losses['loss_rpn_box_reg'] + \
                           losses['loss_classifier'] + losses['loss_box_reg']
            box_features = losses["feats"]
            caption_gt = losses["caption_gt"]
            
            cap = []
            for i in range(len(caption_gt)):
                caption = caption_gt[i].tolist()
                for j, c in enumerate(caption):
                    phrase = ' '.join(idx_to_token[idx] for idx in c if idx_to_token[idx] != '<pad>')
                    phrase = phrase.replace(" <eos>", "")
                    phrase = phrase.replace("<bos> ", "")
                    cap.append(phrase)
            cap_embedding = text_encoder.encode(cap, batch_size=1000)
            cap_embedding = torch.tensor(cap_embedding).cuda()
            
            #loss calculate
            output_proto, target_proto = proto_loss(box_features, cap_embedding, centroid, density)
            criterion = nn.CrossEntropyLoss()
            l_pos = torch.einsum('nc,nc->n', [box_features, cap_embedding]).unsqueeze(-1)
            logits = l_pos
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = criterion(l_pos, labels)
            loss_proto = 0.
            for proto_out,proto_target in zip(output_proto, target_proto):
                loss_proto += criterion(proto_out, proto_target)  
                loss_proto /= len(args.num_cluster) 
                loss += loss_proto
            total_loss = loss * args.proto_loss_weight + detect_loss * args.detect_loss_weight
            
            #back propagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if batch % 1000 == 0:
                print(f"batch:{batch}, total_loss:{total_loss}")

            writer.add_scalar('batch_loss/total', total_loss.item(), iter_counter)
            writer.add_scalar('batch_loss/detect_loss', detect_loss.item(), iter_counter)
            writer.add_scalar('batch_loss/proto_loss', loss.item(), iter_counter)
            #writer.add_scalar('batch_loss/caption_loss', caption_loss.item(), iter_counter)

            writer.add_scalar('details/loss_objectness', losses['loss_objectness'].item(), iter_counter)
            writer.add_scalar('details/loss_rpn_box_reg', losses['loss_rpn_box_reg'].item(), iter_counter)
            writer.add_scalar('details/loss_classifier', losses['loss_classifier'].item(), iter_counter)
            writer.add_scalar('details/loss_box_reg', losses['loss_box_reg'].item(), iter_counter)
            iter_counter += 1

        save_model(visual_encoder, optimizer, iter_counter, flag=str(epoch))
        
    writer.close()
                
def save_model(visual_encoder, optimizer, iter_counter, flag=None):

    visual_state = {'model': visual_encoder.state_dict(),
             'optimizer': optimizer.state_dict(),
             'iterations': iter_counter}

    """text_state = {'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'amp': amp_.state_dict(),
             'iterations': iter_counter}"""

    filename = os.path.join('model_params', '{}_{}.pth.tar'.format(MODEL_NAME, flag))
    print('Saving checkpoint to {}'.format(filename))
    torch.save(visual_state, filename)        

def proto_loss(box_features, cap_features, cluster, distribution):
    proto_labels = []
    proto_logits = []
    r=50
    for n, (prototypes,density) in enumerate(zip(cluster, distribution)):
        prototypes = torch.tensor(prototypes)
        dists = torch.cdist(box_features, prototypes, p=2)
        pos_proto_id = torch.argmax(dists, dim=1)
        pos_prototypes = prototypes[pos_proto_id]    
        
        # sample negative prototypes
        all_proto_id = [i for i in range(prototypes.shape[0])]       
        neg_proto_id = set(all_proto_id)-set(pos_proto_id.tolist())

        neg_proto_id = sample(neg_proto_id,r) #sample r negative prototypes 
        neg_prototypes = prototypes[neg_proto_id]    

        proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
        
        # compute prototypical logits
        logits_proto = torch.mm(box_features,proto_selected.t())
                
        # targets for prototype assignment
        labels_proto = torch.linspace(0, box_features.size(0)-1, steps=box_features.size(0)).long().cuda()
                
        
        # scaling temperatures for the selected prototypes
        temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).cuda()],dim=0)]
        logits_proto /= temp_proto
                
        proto_labels.append(labels_proto)
        proto_logits.append(logits_proto)
    
    return proto_logits, proto_labels
            
    
def sentence_embedding(args, model, sentences, sen_emb_batch_size, k_means_iter, n_cluster):
    print("sentence embedding...")
    embedding_sen = model.encode(sentences, batch_size=sen_emb_batch_size, show_progress_bar=True)
    print("K-means klustering step")
    results = run_kmeans(embedding_sen, args)
    
    return results, embedding_sen


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = args.gpu    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results
    
def main():
    args = set_args()
    #em_train(args)
    caption_train(args)
    

def set_args():
    parser = argparse.ArgumentParser()
    # INPUT settings
    parser.add_argument('--img_root',
                        default='data/visual-genome/VG_100K',
                        help='image path')
    parser.add_argument('--data_path',
                        default='data/VG-regions-lite.h5',
                        help='image path')
    parser.add_argument('--lut_path',
                        default='data/VG-regions-dicts-lite.pkl',
                        help='image path')
    parser.add_argument('--sentences_path',
                        default='data/sentence.pkl',
                        help='sentence file')
    parser.add_argument('--model_path',
                        default='model_params/train_text_freeze_5.pth.tar',
                        help='sentence file')
    
    #model hyperparameter settings
    parser.add_argument('--batch_size',
                        default=4,
                        help='batch_size')
    parser.add_argument('--caption_lr',
                        default=1e-3,
                        help='batch_size')
    parser.add_argument('--lr',
                        default=1e-4,
                        help='batch_size')
    parser.add_argument('--weight_decay',
                        default=0.,
                        help='batch_size')
    parser.add_argument('--proto_loss_weight',
                        default=0.2,
                        help='batch_size')
    parser.add_argument('--detect_loss_weight',
                        default=1,
                        help='batch_size')
    
    
    parser.add_argument('--image_data',
                        default='data/visual-genome/image_data.json',
                        help='Input JSON file with image url weight and height')
    parser.add_argument('--split_json',
                        default='info/densecap_splits.json',
                        help='JSON file of splits')
    
    parser.add_argument('--region_data',
                        default='data/visual-genome/region_descriptions.json',
                        help='Input JSON file with regions and captions')
    
    parser.add_argument('--save_sen_emb',
                        action='store_true',
                        help='store sentence embdeeings vector')
    
    parser.add_argument('--text_encoder',
                        default='paraphrase-MiniLM-L6-v2',
                        help='text embedding model')
    
    parser.add_argument('--num_cluster',
                        default='250', type=str,
                        help='number of k-means cluster')
    
    parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
    
    parser.add_argument('--temperature', default=0.2, type=float,
                    help='softmax temperature')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    main()
