import numpy as np
import pandas as pd
from collections import defaultdict
import peptides
import torch

amino_acid_properties = {
    'A': np.array([1.8, 4.34, 0.0, 0.946, 0.74]),  # Hydrophobicity, Refractivity, Charge, Flexibility, Mean Fractional Area Loss
    'C': np.array([2.5, 35.77, 0.0, 0.878, 0.91]),
    'D': np.array([-3.5, 12.00, -1.0, 1.089, 0.62]),
    'E': np.array([-3.5, 17.26, -1.0, 1.036, 0.62]),
    'F': np.array([2.8, 29.40, 0.0, 0.912, 0.88]),
    'G': np.array([-0.4, 0.00, 0.0, 1.042, 0.72]),
    'H': np.array([-3.2, 21.81, +0.1, 0.952, 0.78]),
    'I': np.array([4.5, 19.06, 0.0, 0.892, 0.88]),
    'K': np.array([-3.9, 21.29, +1.0, 1.082, 0.52]),
    'L': np.array([3.8, 18.78, 0.0, 0.961, 0.85]),
    'M': np.array([1.9, 21.64, 0.0, 0.862, 0.85]),
    'N': np.array([-3.5, 13.28, 0.0, 1.006, 0.63]),
    'P': np.array([-1.6, 10.93, 0.0, 1.085, 0.64]),
    'Q': np.array([-3.5, 17.56, 0.0, 1.025, 0.62]),
    'R': np.array([-4.5, 26.66, +1.0, 1.028, 0.64]),
    'S': np.array([-0.8, 6.35, 0.0, 1.048, 0.66]),
    'T': np.array([-0.7, 11.01, 0.0, 1.051, 0.70]),
    'V': np.array([4.2, 13.92, 0.0, 0.927, 0.86]),
    'W': np.array([-0.9, 42.53, 0.0, 0.917, 0.85]),
    'Y': np.array([-1.3, 31.53, 0.0, 0.930, 0.76]),
}

def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
properties_matrix = np.array(list(amino_acid_properties.values()))
normalized_properties = np.apply_along_axis(min_max_normalization, 0, properties_matrix)
normalized_amino_acid_properties = {
    aa: normalized_properties[i]
    for i, aa in enumerate(amino_acid_properties.keys())
}

queryfile = pd.DataFrame(normalized_amino_acid_properties).T
queryfile.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
def get_interaction_map(cdr3s, peptides, length_cdr3=20, length_peptide=12):  # [20 * 12]
        interaction_map_dict = defaultdict(list)
        features = list(queryfile.columns)
        for idx in range(len(cdr3s)):
                for order in range(5):      # get five interaction_map of one pair
                        interaction_map = np.zeros((length_cdr3, length_peptide))
                        feature =features[order]
                        for i, m in enumerate(cdr3s[idx].upper()):
                                if i >= length_cdr3:
                                        break
                                cdr3_feature = queryfile[feature][m]
                                for j, n in enumerate(peptides[idx].upper()):
                                        if j >= length_peptide:
                                                break
                                        peptide_feature = queryfile[feature][n]
                                        interaction_map[i, j] = abs(cdr3_feature-peptide_feature)
                        interaction_map_dict[feature].append(interaction_map)     # [num, length_cdr3, length_peptide]
        # combine the interaction_maps
        map_0 = np.array(interaction_map_dict[features[0]])
        map_1 = np.array(interaction_map_dict[features[1]])
        map_2 = np.array(interaction_map_dict[features[2]])
        map_3 = np.array(interaction_map_dict[features[3]])
        map_4 = np.array(interaction_map_dict[features[4]])
        combined_map = np.concatenate((np.expand_dims(map_0, 1),
                                       np.expand_dims(map_1, 1),
                                       np.expand_dims(map_2, 1),
                                       np.expand_dims(map_3, 1),
                                       np.expand_dims(map_4, 1)), axis=1)
        return combined_map.tolist()    # height=cdrs, width=peptide

def get_global_attributes_diff(cdr3s, peps):
    diff_vectors = []
    hydrophobicity_table = peptides.tables.HYDROPHOBICITY["KyteDoolittle"]
    for tcr_seq, pep_seq in zip(cdr3s, peps):
        tcr_peptide = peptides.Peptide(tcr_seq)
        peptide_peptide = peptides.Peptide(pep_seq)
        tcr_global_attributes = [
            tcr_peptide.isoelectric_point(),
            tcr_peptide.instability_index(),
            tcr_peptide.aliphatic_index(),
            tcr_peptide.boman(),
            tcr_peptide.hydrophobic_moment(),
            tcr_peptide.molecular_weight(),
            tcr_peptide.auto_correlation(table=hydrophobicity_table, lag=1),
        ]
        peptide_global_attributes = [
            peptide_peptide.isoelectric_point(),
            peptide_peptide.instability_index(),
            peptide_peptide.aliphatic_index(),
            peptide_peptide.boman(),
            peptide_peptide.hydrophobic_moment(),
            peptide_peptide.molecular_weight(),
            peptide_peptide.auto_correlation(table=hydrophobicity_table, lag=1),
        ]
        global_diff_vector = [tcr - pep for tcr, pep in zip(tcr_global_attributes, peptide_global_attributes)]
        diff_vectors.append(global_diff_vector)
    return diff_vectors


# Layer Normalization and Attention Module for Global Features
class GlobalFeatureAttention(torch.nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(feature_dim)  # Layer Normalization
        self.self_attention = torch.nn.MultiheadAttention(embed_dim=feature_dim, num_heads=1)
    def forward(self, x):
        x = x.float()
        x = self.layer_norm(x)
        x = x.unsqueeze(1)
        attention_output, _ = self.self_attention(x, x, x)
        output = attention_output.squeeze(1) + x.squeeze(1)
        # return attention_output.squeeze(1)
        return output
# ResNet
class BasicBlock(torch.nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        identity = x
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(output + identity)
class ResNet(torch.nn.Module):
    def __init__(self, residual, num_residuals, num_classes=2, include_top=True):
        super(ResNet, self).__init__()
        self.out_channel = 64
        self.include_top = include_top
        self.conv1 = torch.nn.Conv2d(5, self.out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channel)
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv2 = self.residual_block(residual, 32, num_residuals[0], stride=1)
        self.conv3 = self.residual_block(residual, 64, num_residuals[1], stride=1)
        self.conv4 = self.residual_block(residual, 128, num_residuals[2], stride=1)
        self.conv5 = self.residual_block(residual, 256, num_residuals[3], stride=1)
        if self.include_top:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            self.fc = torch.nn.Linear(256 * residual.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    def residual_block(self, residual, channel, num_residuals, stride=1):
        downsample = None
        if stride != 1 or self.out_channel != channel * residual.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.out_channel, channel * residual.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(channel * residual.expansion)
            )
        block = [residual(self.out_channel, channel, downsample=downsample, stride=stride)]
        self.out_channel = channel * residual.expansion

        for _ in range(1, num_residuals):
            block.append(residual(self.out_channel, channel))
        return torch.nn.Sequential(*block)
    def forward(self, x):
        output = self.relu(self.bn1(self.conv1(x)))
        output = self.conv5(self.conv4(self.conv3(self.conv2(output))))
        if self.include_top:
            output = self.avgpool(output)
            output = torch.flatten(output, 1)
            output = self.fc(output)
        return output
class SpatialChannelFusion(torch.nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.channel_fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // reduction),
            torch.nn.ReLU(inplace=False),
            torch.nn.Linear(channels // reduction, channels),
            torch.nn.Sigmoid()
        )
        self.spatial_conv = torch.nn.Conv2d(channels, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, resnet_features, global_features):
        batch_size, channels, height, width = resnet_features.size()
        resnet_pooled = self.global_pool(resnet_features).view(batch_size, channels)
        global_pooled = self.global_pool(global_features).view(batch_size, channels)
        channel_weights = self.channel_fc(resnet_pooled + global_pooled).view(batch_size, channels, 1, 1)
        spatial_weights = self.sigmoid(self.spatial_conv(resnet_features + global_features))
        fused_weights = spatial_weights * channel_weights  # [B, C, H, W]
        fused_features = fused_weights * resnet_features + (1 - fused_weights) * global_features
        return fused_features
class SharedConditionAttention(torch.nn.Module):
    def __init__(self, cond_dim, feature_dim, pam_dim):
        super().__init__()
        self.condition_fc_shared = torch.nn.Sequential(
            torch.nn.Linear(cond_dim, cond_dim * 2),
            torch.nn.ReLU(inplace=False)
        )
        self.fc_cam = torch.nn.Linear(cond_dim * 2, feature_dim)
        self.fc_pam = torch.nn.Linear(cond_dim * 2, pam_dim)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, cond_vector, mode="CAM"):
        shared_output = self.condition_fc_shared(cond_vector)
        if mode == "CAM":
            cam_output = self.fc_cam(shared_output)
            return self.sigmoid(cam_output)
        elif mode == "PAM":
            pam_output = self.fc_pam(shared_output)
            return self.sigmoid(pam_output)
        else:
            raise ValueError("Mode must be 'CAM' or 'PAM'")
# PAM module
class PAM_Module(torch.nn.Module):
    """Position Attention Module with Shared Condition"""
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, shared_cond_weight):  # x: [m_batchsize, C, height, width]
        m_batchsize, C, height, width = x.size()
        # proj_query = self.query_conv(x) * shared_cond_weight.view(m_batchsize, C, 1, 1)
        proj_query = self.query_conv(x) * shared_cond_weight.view(m_batchsize, -1, 1, 1)
        proj_query = proj_query.view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out
# CAM module
class CAM_Module(torch.nn.Module):
    """Channel Attention Module with Shared Condition"""
    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x, shared_cond_weight):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out * shared_cond_weight.view(m_batchsize, -1, 1, 1) + x
        return out
# DANetHead 模块
class DANetHead(torch.nn.Module):
    """DANet Head with SharedConditionAttention"""
    def __init__(self, in_channels, out_channels, norm_layer, cond_dim):
        super().__init__()
        inter_channels = in_channels // 4
        self.conv5a = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            torch.nn.ReLU(inplace=False)
        )
        self.sa = PAM_Module(inter_channels)
        self.conv5c = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            torch.nn.ReLU(inplace=False)
        )
        self.sc = CAM_Module(inter_channels)
        self.shared_condition = SharedConditionAttention(cond_dim, inter_channels, inter_channels // 8)
        self.conv51 = torch.nn.Sequential(
            torch.nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            torch.nn.ReLU(inplace=False)
        )
        self.conv52 = torch.nn.Sequential(
            torch.nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            torch.nn.ReLU(inplace=False)
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(inter_channels, out_channels, 1)
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(inter_channels, out_channels, 1)
        )
        self.conv8 = torch.nn.Sequential(
            torch.nn.Dropout2d(0.3),
            torch.nn.Conv2d(inter_channels, out_channels, 1)
        )
    def forward(self, x, cond_vector):
        shared_cond_weight_pam = self.shared_condition(cond_vector, mode="PAM")
        shared_cond_weight_cam = self.shared_condition(cond_vector, mode="CAM")
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1, shared_cond_weight_pam)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2, shared_cond_weight_cam)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)
        feat_sum = sa_conv + sc_conv
        sasc_output = self.conv8(feat_sum)
        return sasc_output, sa_output, sc_output
class ResNetDANet(torch.nn.Module):
    def __init__(self, num_classes=2, global_feature_dim=7):
        super().__init__()
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=False)
        self.global_attention = GlobalFeatureAttention(global_feature_dim)
        self.danet_head = DANetHead(256, num_classes, norm_layer=torch.nn.BatchNorm2d, cond_dim=global_feature_dim)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_channel_fusion = SpatialChannelFusion(256)
        self.global_transform = torch.nn.Conv2d(global_feature_dim, 256, kernel_size=1)
    def forward(self, x, global_diff_vector):  # x:[batch_size, 5, 20, 12]  global_diff_vector:[batch_size, global_feature_dim]
        resnet_features = self.resnet(x)
        global_features = self.global_attention(global_diff_vector)
        global_features_copy = global_features.clone()
        global_features_expanded = global_features_copy.unsqueeze(-1).unsqueeze(-1)  # [batch_size, global_feature_dim, 1, 1]
        global_features_expanded = self.global_transform(global_features_expanded)
        global_features_expanded = global_features_expanded.expand(-1, -1, resnet_features.size(2), resnet_features.size(3))

        fused_features = self.spatial_channel_fusion(resnet_features, global_features_expanded)
        danet_output = self.danet_head(fused_features, global_features)
        out = danet_output[0]
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
def resnet_danet(num_classes=2, global_feature_dim=7):
    return ResNetDANet(num_classes=num_classes, global_feature_dim=global_feature_dim)
