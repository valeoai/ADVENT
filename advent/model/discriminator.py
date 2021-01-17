from torch import nn


def get_fc_discriminator(num_classes, ndf=64):
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )


# def get_fe_discriminator(num_classes, ndf=64): # 256-128-64-32-16
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf, kernel_size=2, stride=2, padding=0),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         # nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#         # nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, 1, kernel_size=2, stride=2, padding=0),
#     )

# def get_fe_discriminator(num_classes, ndf=64):
#     return nn.Sequential(
#         nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         # nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
#         # nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(ndf * 4, 1, kernel_size=1, stride=1, padding=0),
#     )

def get_fe_discriminator(num_classes, ndf=64):  # H/8,H/8,(1024 -> 256 -> 128 -> 64  -> 1)
    return nn.Sequential(
        nn.Conv2d(num_classes, ndf * 4, kernel_size=1, stride=1, padding=0),
        # x=self.dropout(x)
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 2, kernel_size=1, stride=1, padding=0),
        # x=self.dropout(x)
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf, kernel_size=1, stride=1, padding=0),
        # x=self.dropout(x)
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        # nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, 1, kernel_size=1, stride=1, padding=0),
    )