import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

import tensorflow as tf
import keras
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC
from seq2tensor import s2t
from model import*
from load_and_preprocess_data import*

def setup_gpu(gpu_ids="1,2"):
    """配置GPU设置"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    print("TensorFlow is using GPU:", tf.config.list_logical_devices('GPU'))
    gpus = tf.config.list_physical_devices('GPU')
    print("Available GPUs:", gpus)


def parse_arguments():

    parser = argparse.ArgumentParser(description='Protein Interaction Prediction Model')
    

    parser.add_argument('--positive_data', type=str, 
                       default='process_data/Yeast_data/positive_protein.xlsx',
                       help='Path to positive protein data')
    parser.add_argument('--negative_data', type=str,
                       default='process_data/Yeast_data/negative_protein.xlsx', 
                       help='Path to negative protein data')
    parser.add_argument('--max_length', type=int, default=2000,
                       help='Sequence size for padding/truncation')
    parser.add_argument('--use_emb', type=int, default=3)
    

    parser.add_argument('--hidden_dim', type=int, default=50,
                       help='Hidden dimension size')
    # parser.add_argument('--dff', type=int, default=2048,
    #                    help='Feed-forward dimension')
    parser.add_argument('--num_heads', type=int, default=1,
                       help='Number of attention heads')
    parser.add_argument('--embed_dim', type=int, default=13)
    

    parser.add_argument('--n_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--l2_reg', type=float, default=1e-5,
                       help='L2 regularization strength')
    

    parser.add_argument('--rst_file', type=str, 
                       default='result/Yeast/results.txt',
                       help='Results output file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default='result/Yeast/save_model/',
                       help='Model checkpoint directory')
    parser.add_argument('--roc_curve_path', type=str,
                       default='result/Yeast/visualization/roc_curve.png',
                       help='ROC curve output path')
    

    parser.add_argument('--train_seed', type=int, default=42,
                       help='Random seed for train/val split')
    parser.add_argument('--val_seed', type=int, default=42,
                       help='Random seed for val/test split')
    
    return parser.parse_args()


def split_data(class_labels, train_seed, val_seed):
    all_indices = np.arange(len(class_labels))
    
    train_idx, temp_idx = train_test_split(
        all_indices,
        test_size=0.2,
        stratify=class_labels,
        shuffle=True,
        random_state=train_seed
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=class_labels[temp_idx],
        shuffle=True,
        random_state=val_seed
    )
    
    assert len(set(train_idx).intersection(set(test_idx))) == 0
    assert len(set(train_idx).intersection(set(val_idx))) == 0
    assert len(set(val_idx).intersection(set(test_idx))) == 0
    
    print(f"Train: {len(train_idx)}, Validation: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx

def plot_roc_curve(fpr, tpr, roc_auc, save_path):

    if fpr is None or tpr is None:
        print("Cannot plot ROC curve due to evaluation issues")
        return
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def save_results(metrics, rst_file):

    os.makedirs(os.path.dirname(rst_file), exist_ok=True)
    
    if not os.path.exists(rst_file):
        with open(rst_file, 'w') as f:
            f.write('Epoch\tACC_test\tPrecision_test\tRecall_test\tAUC_test\n')
    
    with open(rst_file, 'a') as f:
        f.write(f"{metrics['accuracy']:.4f}\t{metrics['precision']:.4f}\t"
                f"{metrics['recall']:.4f}\t{metrics['auc']:.4f}\n")


def main():

    args = parse_arguments()
    tokenizer = create_tokenizer_custom(file='tokenizer.json')
    vocab_size = tokenizer.get_vocab_size()

    emb_files = ['./embeddings/default_onehot.txt', './embeddings/string_vec5.txt', './embeddings/CTCoding_onehot.txt', './embeddings/vec5_CTC.txt']
    seq2t = s2t(emb_files[args.use_emb])

    setup_gpu()
    
    seq_tensor, seq_index1, seq_index2, class_labels = load_and_preprocess_data_seq2t(
        args.positive_data, args.negative_data, args.max_length,seq2t
    )
    
    train_idx, val_idx, test_idx = split_data(
        class_labels, args.train_seed, args.val_seed
    )

    # model = build_model_wte(args.max_length, vocab_size,args.embed_dim, args.hidden_dim, 
    #                    l2(args.l2_reg),args.num_heads)
    dim = seq2t.dim
    model = build_model(args.max_length,dim, args.hidden_dim, 
                       l2(args.l2_reg),args.num_heads)
    
    print(f"Model built with {model.count_params()} parameters")
    
    print("Starting training...")
    

    optimizer = Adam(learning_rate=args.learning_rate, amsgrad=True, epsilon=1e-6)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', AUC(name='auc')]
    )
    
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f"rcnn_attention_{args.n_epochs}_lr{args.learning_rate}_dim{args.hidden_dim}.keras"
    )
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    

    history = model.fit(
        [seq_tensor[seq_index1[train_idx]], seq_tensor[seq_index2[train_idx]]],
        class_labels[train_idx],
        validation_data=(
            [seq_tensor[seq_index1[val_idx]], seq_tensor[seq_index2[val_idx]]],
            class_labels[val_idx]),
        batch_size=args.batch_size,
        epochs=args.n_epochs,
        callbacks=[model_checkpoint],
        shuffle=False
    )

    model.load_weights(checkpoint_path)

    print("Evaluating model...")
    pred = model.predict([seq_tensor[seq_index1[test_idx]], seq_tensor[seq_index2[test_idx]]])
    
    true_labels = np.argmax(class_labels[test_idx], axis=1)
    pred_labels = np.argmax(pred, axis=1)
    positive_probs = pred[:, 0]
    
    num_total = len(test_idx)
    num_hit = np.sum(true_labels == pred_labels)
    
    num_pos = np.sum(class_labels[test_idx][:, 0] > 0.)
    num_true_pos = np.sum((true_labels == 0) & (pred_labels == 0))
    num_false_pos = np.sum((true_labels == 1) & (pred_labels == 0))
    num_true_neg = np.sum((true_labels == 1) & (pred_labels == 1))
    num_false_neg = np.sum((true_labels == 0) & (pred_labels == 1))
    
    accuracy = num_hit / num_total
    precision = num_true_pos / (num_true_pos + num_false_pos) if (num_true_pos + num_false_pos) > 0 else 0
    recall = num_true_pos / num_pos if num_pos > 0 else 0
    specificity = num_true_neg / (num_true_neg + num_false_neg) if (num_true_neg + num_false_neg) > 0 else 0
    f1 = 2. * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    try:
        auc_score = roc_auc_score(true_labels, 1 - positive_probs)
        fpr, tpr, _ = roc_curve(true_labels, positive_probs, pos_label=0)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        auc_score = float('nan')
        roc_auc = float('nan')
        fpr, tpr = None, None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': roc_auc
    }
    
    print(f"Results:")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    
    plot_roc_curve(fpr, tpr, metrics['auc'], args.roc_curve_path)
    
    save_results(metrics, args.rst_file)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()