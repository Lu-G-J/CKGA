import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, matthews_corrcoef
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from datasets import my_collate, my_collate_elmo, my_collate_pure_bert, my_collate_bert
from transformers import AdamW
from transformers import BertTokenizer

import pickle

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(args, batch):
    embedding_type = args.embedding_type
    if embedding_type == 'glove' or embedding_type == 'elmo':
        # sentence_ids, aspect_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
        inputs = {  'sentence': batch[0],
                    'aspect': batch[1], # aspect token
                    'dep_tags': batch[2], # reshaped
                    'pos_class': batch[3],
                    'text_len': batch[4],
                    'aspect_len': batch[5],
                    'dep_rels': batch[7], # adj no-reshape
                    'dep_heads': batch[8],
                    'aspect_position': batch[9],
                    'dep_dirs': batch[10],
                  ################## changed
                    'entity_idx': batch[11]
                  ################## end
                    }
        labels = batch[6]
    else: # bert
        if args.pure_bert:
            # input_cat_ids, segment_ids, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'token_type_ids': batch[1]}
            labels = batch[6]
        else:
            # input_ids, word_indexer, input_aspect_ids, aspect_indexer, dep_tag_ids, pos_class, text_len, aspect_len, sentiment, dep_rel_ids, dep_heads, aspect_positions
            inputs = {  'input_ids': batch[0],
                        'input_aspect_ids': batch[2],
                        'word_indexer': batch[1],
                        'aspect_indexer': batch[3],
                        'input_cat_ids': batch[4],
                        'segment_ids': batch[5],
                        'dep_tags': batch[6],
                        'pos_class': batch[7],
                        'text_len': batch[8],
                        'aspect_len': batch[9],
                        'dep_rels': batch[11],
                        'dep_heads': batch[12],
                        'aspect_position': batch[13],
                        'dep_dirs': batch[14],
                      'entity_idx': batch[15]}
            labels = batch[10]
    return inputs, labels


def get_collate_fn(args):
    embedding_type = args.embedding_type
    if embedding_type == 'glove':
        return my_collate
    elif embedding_type == 'elmo':
        return my_collate_elmo
    else:
        if args.pure_bert:
            return my_collate_pure_bert
        else:
            return my_collate_bert


def get_bert_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    return optimizer


def train(args, train_dataset, model, test_dataset):
    '''Train the model'''
    tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    collate_fn = get_collate_fn(args)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    
    if args.embedding_type == 'bert':
        optimizer = get_bert_optimizer(args, model)
    else:
        parameters = filter(lambda param: param.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        #optimizer = torch.optim.Adam(parameters, lr=args.origin_model_lr)
        
    ################################# started
    if args.train_model == 'd':
        model.origin_model.load_state_dict(torch.load(args.origin_model_path))
    params1 = [
        {"params": [p for p in model.origin_model.parameters() if p.requires_grad], "lr":args.origin_model_lr},
        {"params": [p for p in model.classification.parameters() if p.requires_grad], "lr":args.learning_rate}, 
        {"params": [p for p in model.mha.parameters() if p.requires_grad], "lr":args.learning_rate}, 
    ]
    params2 = [
        {"params": [p for p in model.adapter.parameters() if p.requires_grad], "lr":args.learning_rate}
    ]
    optimizer1 = torch.optim.Adam(params1, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(params2, lr=args.learning_rate, weight_decay=args.weight_decay)    
        
    
    deleted_path = 'init'
    deleted_arg_path = 'init'
    ################################# end
    

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    all_eval_results = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)
    ###################### changed
    if not os.path.exists('state_dict/args'):
        #os.mkdir('state_dict')
        os.makedirs('state_dict/args')
    
    for idx, _ in enumerate(train_iterator):
        #if idx > 5:
        #    args.logging_steps = 10
        # epoch_iterator = tqdm(train_dataloader, desc='Iteration')
        
        
        optimizer2.zero_grad()
        model.adapter(None, gcn=True)
    ######################
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            ################################################### changed
            inputs, labels = get_input_from_batch(args, batch)
            adapter_inputs = inputs.pop('entity_idx')
            origin_model_inputs = inputs
            
            #logit = model(**inputs)
            logit = model(origin_model_inputs, adapter_inputs, model=args.adapter_mode, mode=args.fuse_mode,input_mode='dict')
            ################################################### 
            loss = F.cross_entropy(logit, labels)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            ###################### changed
            loss.backward(retain_graph=True)
            ######################
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            tr_loss = tr_loss + loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # scheduler.step()  # Update learning rate schedule
                optimizer1.step()
                optimizer1.zero_grad() # changed
                #model.zero_grad()
                global_step = global_step + 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    results, eval_loss = evaluate(args, test_dataset, model)
                    all_eval_results.append(results)
                    
                    #################################### start
                    
                    
                    
                    best_eval_result = max(all_eval_results, key=lambda x: x['acc'])
                    if results['acc'] == best_eval_result['acc']:
                        '''
                        torch.save(model.state_dict(), 'state_dict/'+args.dataset_name+'_'+args.embedding_type+'_acc_'+str('%.4f'%results['acc'])+'_f1_'+str('%.4f'%results['f1'])+'.pkl')
                        pickle.dump(args, open('state_dict/'+args.dataset_name+'_'+args.embedding_type+'_acc_'+str('%.4f'%results['acc'])+'_f1_'+str('%.4f'%results['f1'])+'_arg.pkl','wb'))
                        deleted_path.append( 'state_dict/'+args.dataset_name+'_'+args.embedding_type+'_acc_'+str('%.4f'%results['acc'])+'_f1_'+str('%.4f'%results['f1'])+'.pkl') 
                        deleted_path.append( 'state_dict/'+args.dataset_name+'_'+args.embedding_type+'_acc_'+str('%.4f'%results['acc'])+'_f1_'+str('%.4f'%results['f1'])+'_arg.pkl')
                        '''
                        if deleted_path != 'init' and deleted_arg_path != 'init':
                            os.remove(deleted_path)
                            os.remove(deleted_arg_path)
                        deleted_path = f"state_dict/{args.dataset_name}_{args.embedding_type}_acc_{'%.4f'%results['acc']}_f1_{'%.4f'%results['f1']}.pkl"
                        deleted_arg_path = f"state_dict/args/{args.dataset_name}_{args.embedding_type}_acc_{'%.4f'%results['acc']}_f1_{'%.4f'%results['f1']}_arg.pkl"
                        torch.save(model.state_dict(), deleted_path)
                        pickle.dump(args, open(deleted_arg_path,'wb'))
                        
                    #################################### end
                    
                    
                    
                    for key, value in results.items():
                        tb_writer.add_scalar(
                            'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('eval_loss', eval_loss, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'train_loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        ###################### changed
        optimizer2.step() # changed
        ######################
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
    ############################ start
    '''
    for path in deleted_path:
        if path != f"state_dict/{args.dataset_name}_{args.embedding_type}_{args.adapter_kge}_{args.train_model}_{args.origin_model_lr}_acc_{'%.4f'%best_eval_result['acc']}_f1_{'%.4f'%best_eval_result['f1']}.pkl" and path != f"state_dict/args/{args.dataset_name}_{args.embedding_type}_{args.adapter_kge}_{args.train_model}_{args.origin_model_lr}_acc_{'%.4f'%best_eval_result['acc']}_f1_{'%.4f'%best_eval_result['f1']}_arg.pkl":
            os.remove(path)
    '''
    ############################ end
    
    
    tb_writer.close()
    return global_step, tr_loss/global_step, all_eval_results


def evaluate(args, eval_dataset, model):
    results = {}

    args.eval_batch_size = args.per_gpu_eval_batch_size
    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 collate_fn=collate_fn)

    # Eval
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            ######################## changed
            inputs, labels = get_input_from_batch(args, batch)
            
            adapter_inputs = inputs.pop('entity_idx')
            origin_model_inputs = inputs
            
            #logit = model(**inputs)
            logits = model(origin_model_inputs, adapter_inputs, model=args.adapter_mode, mode=args.fuse_mode,input_mode='dict')
            
            ######################## end
            tmp_eval_loss = F.cross_entropy(logits, labels)

            eval_loss = eval_loss + tmp_eval_loss.mean().item()
        nb_eval_steps = nb_eval_steps + 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    # print(preds)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)
    
    output_eval_file = os.path.join(args.output_dir, 'eval_results.txt')
    with open(output_eval_file, 'a+') as writer:
        logger.info('***** Eval results *****')
        logger.info("  eval loss: %s", str(eval_loss))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("  %s = %s\n" % (key, str(result[key])))
            writer.write('\n')
        writer.write('\n')
    return results, eval_loss


def evaluate_badcase(args, eval_dataset, model, word_vocab):

    eval_sampler = SequentialSampler(eval_dataset)
    collate_fn = get_collate_fn(args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=1,
                                 collate_fn=collate_fn)

    # Eval
    badcases = []
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
    # for batch in tqdm(eval_dataloader, desc='Evaluating'):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs, labels = get_input_from_batch(args, batch)

            logits = model(**inputs)

        pred = int(np.argmax(logits.detach().cpu().numpy(), axis=1)[0])
        label = int(labels.detach().cpu().numpy()[0])
        if pred != label:
            if args.embedding_type == 'bert':
                sent_ids = inputs['input_ids'][0].detach().cpu().numpy()
                aspect_ids = inputs['input_aspect_ids'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = args.tokenizer.decode(sent_ids)
                case['aspect'] = args.tokenizer.decode(aspect_ids)
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)
            else:
                sent_ids = inputs['sentence'][0].detach().cpu().numpy()
                aspect_ids = inputs['aspect'][0].detach().cpu().numpy()
                case = {}
                case['sentence'] = ' '.join([word_vocab['itos'][i] for i in sent_ids])
                case['aspect'] = ' '.join([word_vocab['itos'][i] for i in aspect_ids])
                case['pred'] = pred
                case['label'] = label
                badcases.append(case)

    return badcases


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1
    }


def compute_metrics(preds, labels):
    return acc_and_f1(preds, labels)
