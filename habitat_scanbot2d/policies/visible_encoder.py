import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict


class VisibleEncoder(torch.nn.Module):
    def __init__(
        self,
        base_planes=32,
        global_planes=64,
        num_views=8 * 3,
    ):
        """
        Encode the global voxels into a global feature and concatenate it with features_list, a fc_layer is followed to
        obtain the visibility of each view. (The visibility of each view is represented by two dimensions of the output.)
        """
        super(VisibleEncoder, self).__init__()
        self.num_classes = num_views
        self.global_planes = global_planes
        self.main = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv3d(
                            in_channels=1,
                            out_channels=base_planes,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("leakyrelu1", nn.LeakyReLU()),
                    ("maxpool1", nn.MaxPool3d(2)),
                    (
                        "conv2",
                        nn.Conv3d(
                            in_channels=base_planes,
                            out_channels=base_planes * 2,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("leakyrelu2", nn.LeakyReLU()),
                    ("maxpool2", nn.MaxPool3d(2)),
                    (
                        "conv3",
                        nn.Conv3d(
                            in_channels=base_planes * 2,
                            out_channels=base_planes * 2 * 2,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("leakyrelu3", nn.LeakyReLU()),
                    ("maxpool3", nn.MaxPool3d(2)),
                    (
                        "conv4",
                        nn.Conv3d(
                            in_channels=base_planes * 2 * 2,
                            out_channels=base_planes * 2,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("leakyrelu4", nn.LeakyReLU()),
                    ("maxpool4", nn.MaxPool3d(2)),
                    (
                        "conv5",
                        nn.Conv3d(
                            in_channels=base_planes * 2,
                            out_channels=base_planes,
                            kernel_size=3,
                            padding=1,
                        ),
                    ),
                    ("leakyrelu5", nn.LeakyReLU()),
                    ("flatten", nn.Flatten()),
                ]
            )
        )

    def fc_layer(self, in_planes, hidden_size=256, final_fc=False):
        if not final_fc:
            fc = nn.Sequential(
                nn.Linear(in_planes, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, self.global_planes),
                nn.ReLU(),
            ).cuda()
        else:
            fc = nn.Sequential(
                nn.Linear(in_planes, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size, hidden_size // 16),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(hidden_size // 16, self.num_classes),
            ).cuda()
        return fc

    def forward(self, global_voxels, feature_list=None, device = 'cpu'):
        global_voxels = torch.from_numpy(global_voxels[np.newaxis, np.newaxis,...]).float().to(device)
        global_features = self.main(global_voxels)
        global_features = self.fc_layer(in_planes=global_features.size(1))(
            global_features
        )
        x = (
            torch.cat([global_features, feature_list], dim=1)
            if feature_list is not None
            else global_features
        )
        x = self.fc_layer(x.size(1), final_fc=True)(x)
        return np.squeeze(x.cpu().numpy())


if __name__ == "__main__":
    # torch.cuda.set_device('cuda:0')
    num_views = 8 * 3
    x = torch.rand([32, 1, 64, 64, 64]).cuda()

    ## encode and concatenate
    model = VisibleEncoder(
        global_planes=64,
        num_views=num_views,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    optimizer.zero_grad()

    x = model(x, feature_list=None)
    x = nn.Softmax(dim=1)(x.view(-1, 2))

    labels = (
        torch.randint(0, 2, [32, num_views])
        .view(
            -1,
        )
        .cuda()
    )
    criterion = nn.CrossEntropyLoss()
    loss = criterion(x, labels)
    loss.backward()
    optimizer.step()
