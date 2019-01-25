"""Main script for bert-DANN."""

from params import param
from core import train_src, eval_tgt
from models import BERTEncoder, BERTClassifier, DomainClassifier
from utils import read_data, get_data_loader, init_model, init_random_seed
from pytorch_pretrained_bert import BertTokenizer
import argparse
import numpy as np
import torch

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Specify Params for Experimental Setting")

    parser.add_argument('--src', type=str, default="books", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify src dataset")

    parser.add_argument('--tgt', type=str, default="dvd", choices=["books", "dvd", "electronics", "kitchen"],
                        help="Specify tgt dataset")

    parser.add_argument('--seqlen', type=int, default=50,
                        help="Specify maximum sequence length")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Specify batch size")

    parser.add_argument('--domain_weight', type=float, default=0.05,
                        help="Specify domain weight")

    parser.add_argument('--num_epochs', type=int, default=5,
                        help="Specify the number of epochs for training")

    parser.add_argument('--log_step', type=int, default=1,
                        help="Specify log step size for training")

    parser.add_argument('--eval_step', type=int, default=1,
                        help="Specify eval step size for training")

    parser.add_argument('--save_step', type=int, default=100,
                        help="Specify save step size for training")

    args = parser.parse_args()

    # argument setting
    print("=== Argument Setting ===")
    print("src: " + args.src)
    print("tgt: " + args.tgt)
    print("seqlen: " + str(args.seqlen))
    print("batch_size: " + str(args.batch_size))
    print("domain_weight: " + str(args.domain_weight))
    print("num_epochs: " + str(args.num_epochs))
    print("log_step: " + str(args.log_step))
    print("eval_step: " + str(args.eval_step))
    print("save_step: " + str(args.save_step))

    # preprocess data
    print("=== Processing datasets ===")
    reviews, labels = XML2Array(os.path.join('data', args.src, 'negative.parsed'),
                                os.path.join('data', args.src, 'positive.parsed'))

    src_X_train, src_X_test, src_Y_train, src_Y_test = train_test_split(reviews, labels,
                                                                        test_size=0.2,
                                                                        random_state=args.random_state)
    del reviews, labels

    if args.tgt == 'blog':
        tgt_X, tgt_Y = blog2Array(os.path.join('data', args.tgt, 'blog.parsed'))

    else:
        tgt_X, tgt_Y = XML2Array(os.path.join('data', args.tgt, 'negative.parsed'),
                                 os.path.join('data', args.tgt, 'positive.parsed'))

    src_X_train = review2seq(src_X_train)
    src_X_test = review2seq(src_X_test)
    tgt_X = review2seq(tgt_X)

    # load dataset
    src_data_loader = get_data_loader(src_X_train, src_Y_train, args.batch_size, args.seqlen)
    src_data_loader_eval = get_data_loader(src_X_test, src_Y_test, args.batch_size, args.seqlen)
    tgt_data_loader = get_data_loader(tgt_X, tgt_Y, args.batch_size, args.seqlen)

    # load models
    encoder = BERTEncoder()
    class_classifier = BERTClassifier()
    domain_classifier = DomainClassifier()

    if torch.cuda.device_count() > 1:
        encoder = torch.nn.DataParallel(encoder)
        class_classifier = torch.nn.DataParallel(class_classifier)
        domain_encoder = torch.nn.DataParallel(domain_classifier)

    encoder = init_model(encoder,
                         restore=param.encoder_restore)
    class_classifier = init_model(class_classifier,
                                  restore=param.class_classifier_restore)
    domain_classifier = init_model(domain_classifier,
                                   restore=param.domain_classifier_restore)

    # freeze source encoder params
    if torch.cuda.device_count() > 1:
        for params in encoder.module.encoder.embeddings.parameters():
            params.requires_grad = False
    else:
        for params in encoder.encoder.embeddings.parameters():
            params.requires_grad = False

    # train source model
    print("=== Training classifier for source domain ===")
    src_encoder, class_classifier, domain_classifier = train_src(
        args, encoder, class_classifier, domain_classifier, src_data_loader, tgt_data_loader, src_data_loader_eval)

    # eval target encoder on lambda0.1 set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> DANN adaption <<<")
    eval_tgt(encoder, class_classifier, tgt_data_loader)
