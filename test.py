import argparse
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, precision_recall_curve, auc
from DAISY import resnet_danet, get_global_attributes_diff, get_interaction_map

parser = argparse.ArgumentParser(description='predict whether the tcr and peptide can bind')
parser.add_argument('--input_dir', type=str, default='./=data/test_data', help='directory containing multiple input test files (CSV format)')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for prediction')
parser.add_argument('--tcr_peptide_model', type=str, default='./model/DAISY.pt', help='tcr_peptide prediction model file')
parser.add_argument('--output_dir', type=str, default='./output/', help='directory to save prediction results')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class TCRPeptideDataset(Dataset):
    def __init__(self, cdr3s, peptides, labels):
        self.cdr3s = cdr3s
        self.peptides = peptides
        self.labels = labels

    def __len__(self):
        return len(self.cdr3s)

    def __getitem__(self, idx):
        return self.cdr3s[idx], self.peptides[idx], self.labels[idx]


model = resnet_danet()
model.load_state_dict(torch.load(args.tcr_pmhc_model))
model.to(DEVICE)
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
model.eval()


def evaluate(model, test_loader):
    probs, preds, test_labels = [], [], []
    with torch.no_grad():
        for batch_cdr3s, batch_peptides, batch_labels in test_loader:

            batch_maps = torch.tensor(get_interaction_map(batch_cdr3s, batch_peptides)).to(DEVICE)
            batch_diffs = torch.tensor(get_global_attributes_diff(batch_cdr3s, batch_peptides)).to(DEVICE)

            outputs = model(batch_maps, batch_diffs)  # [batch_size, 2]
            batch_probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
            batch_preds = [1 if prob > 0.5 else 0 for prob in batch_probs]

            probs.extend(batch_probs)
            preds.extend(batch_preds)
            test_labels.extend(batch_labels.tolist())

    ACC = accuracy_score(test_labels, preds)
    ROC_AUC = roc_auc_score(test_labels, probs)
    Recall = recall_score(test_labels, preds)
    Precision = precision_score(test_labels, preds)
    F1 = f1_score(test_labels, preds)
    precision, recall, _ = precision_recall_curve(test_labels, probs)
    PR_AUC = auc(recall, precision)
    return probs, preds, ACC, ROC_AUC, Recall, Precision, F1, PR_AUC


test_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
results_summary = []
for test_file in test_files:
    test_path = os.path.join(args.input_dir, test_file)
    input_df = pd.read_csv(test_path)
    cdr3s = input_df['cdr3'].tolist()
    peptides = input_df['peptide'].tolist()
    labels = input_df['label'].tolist()
    dataset = TCRPeptideDataset(cdr3s, peptides, labels)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    probs, preds, ACC, ROC_AUC, Recall, Precision, F1, PR_AUC = evaluate(model, test_loader)

    input_df['probs'] = probs
    input_df['preds'] = preds
    output_path = os.path.join(args.output_dir, f'predicted_{test_file}')
    input_df.to_csv(output_path, index=False)

    results_summary.append({
        'file_name': test_file,
        'ACC': ACC,
        'ROC_AUC': ROC_AUC,
        'Recall': Recall,
        'Precision': Precision,
        'F1': F1,
        'PR_AUC': PR_AUC
    })
    print(f'Processed {test_file} - ACC: {ACC:.4f}, ROC_AUC: {ROC_AUC:.4f}, Recall: {Recall:.4f}, Precision: {Precision:.4f}, F1: {F1:.4f}, PR_AUC: {PR_AUC:.4f}')

summary_df = pd.DataFrame(results_summary)
summary_df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)
print(f'Results summary saved to {os.path.join(args.output_dir, "results_summary.csv")}')
print('test is end')