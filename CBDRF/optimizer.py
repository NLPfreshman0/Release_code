import torch
from transformers import get_linear_schedule_with_warmup

def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
         'weight_decay': args.weight_decay,
         'lr': args.learning_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n], 
         'weight_decay': 0.0,
         'lr': args.learning_rate}
    ]
    
    optimizer_grouped_parameters += [
        {"params": [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay) and 'bert' not in n)],
            "weight_decay": args.weight_decay,
            "lr": 1e-4,
        },
        {"params": [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay) and 'bert' not in n)],
            "weight_decay": 0.0,
            "lr": 1e-4,
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler
