import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from .contrast_loss import ContrastLoss
from .deform_conv import DeformableConvRGB


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, degradation_kernel, index_1):
        b, c = degradation_kernel.shape

        degradation_kernel_exp = degradation_kernel[self._batch_index]

        list1 = torch.zeros((1, self._num_experts))
        list1[0, index_1] = b

        return torch.split(degradation_kernel_exp, list1[0].int().tolist(), dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates.unsqueeze(1).unsqueeze(1))

        zeros = torch.zeros(
            (self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3)),
            requires_grad=True,
            device=stitched.device,
        )

        combined = zeros.index_add(0, self._batch_index, stitched.float())

        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class DecoderMixtureOfExperts(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(
        self, ds_inputsize, input_size, output_size, num_experts, hidden_size, noisy_gating=True, k=2, trainingmode=True
    ):
        super(DecoderMixtureOfExperts, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.training = trainingmode
        self.k = k
        # instantiate experts
        self.experts = nn.ModuleList(
            [
                DegradationKernelGenerator(hidden_size, 3),
                DegradationKernelGenerator(hidden_size, 5),
                DegradationKernelGenerator(hidden_size, 7),
                DegradationKernelGenerator(hidden_size, 9),
            ]
        )
        self.w_gate = nn.Parameter(torch.zeros(ds_inputsize, num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(ds_inputsize, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: https://arxiv.org/abs/1701.06538.
        Args:
          x: input Tensor with shape [batch_size, input_size]
          train: a boolean - we only add noise at training time.
          noise_epsilon: a float
        Returns:
          gates: a Tensor with shape [batch_size, num_experts]
          load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load, top_k_indices[0]

    def forward(self, x_ds, degradation_kernel, loss_coef=1e-2):
        gates, load, index_1 = self.noisy_top_k_gating(x_ds, self.training)
        # calculate importance loss
        importance = gates.sum(0)

        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_kernel = dispatcher.dispatch(degradation_kernel, index_1)
        expert_outputs = [self.experts[i](expert_kernel[i]) for i in range(self.num_experts)]
        return expert_outputs, loss


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResidualChannelAttentionBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(ChannelAttention(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            ResidualChannelAttentionBlock(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.LeakyReLU(negative_slope=0.2, inplace=True),
                res_scale=1,
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))


class DepthEncoder(nn.Module):
    def __init__(self, n_feats):
        super(DepthEncoder, self).__init__()

        self.E_pre = nn.Sequential(
            ResBlock(in_feat=1, out_feat=n_feats // 2, stride=1),
            ResBlock(in_feat=n_feats // 2, out_feat=n_feats, stride=1),
            ResBlock(in_feat=n_feats, out_feat=n_feats, stride=1),
        )
        self.E = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_feats * 2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_feats * 4),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter)

        out = fea.squeeze(-1).squeeze(-1)

        return fea, out, inter


class DegradationKernelGenerator(nn.Module):
    def __init__(self, n_feats, kernel_size=5):
        super(DegradationKernelGenerator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats), nn.LeakyReLU(0.1, True), nn.Linear(n_feats, kernel_size * kernel_size)
        )

    def forward(self, degradation_kernel):
        degradation_kernel = self.mlp(degradation_kernel)
        return degradation_kernel


class DegradationFilter(nn.Module):
    def __init__(self):
        super(DegradationFilter, self).__init__()
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv = default_conv(1, 1, 1)

    def forward(self, x, degradation_kernel):
        batch_size, channel_size, height, width = x.size()
        latent_batch_size, latent_dim = degradation_kernel.shape
        kernel_size = int(math.sqrt(latent_dim))
        with torch.no_grad():
            kernel = degradation_kernel.view(-1, 1, kernel_size, kernel_size)
            out = F.conv2d(
                x.view(1, -1, height, width), kernel, groups=batch_size * channel_size, padding=(kernel_size - 1) // 2
            )
            out = out.view(batch_size, -1, height, width)
        out = self.conv(self.relu(out).view(batch_size, -1, height, width))
        return out


class DegradationRouter(nn.Module):
    def __init__(self, n_feats, num_experts=4, k=3):
        super(DegradationRouter, self).__init__()

        self.topK = k
        self.num_experts = num_experts
        self.start_idx = num_experts - k

        self.c1 = ResBlock(in_feat=1, out_feat=n_feats, stride=1)
        self.gap = nn.AdaptiveMaxPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(n_feats, n_feats * 4)

        degradation_filters = [DegradationFilter(), DegradationFilter(), DegradationFilter()]
        self.degradation_filters = nn.ModuleList(degradation_filters)

        self.decoder_moe = DecoderMixtureOfExperts(
            ds_inputsize=n_feats * 4,
            input_size=1,
            output_size=1,
            num_experts=num_experts,
            hidden_size=n_feats,
            noisy_gating=True,
            k=k,
            trainingmode=True,
        )

        self.conv = default_conv(1, 1, 1)

    def forward(self, lr, sr, degradation_kernel):

        y1 = F.interpolate(lr, scale_factor=0.125, mode='bicubic', align_corners=True, recompute_scale_factor=True)
        y2 = self.c1(y1)
        y3 = self.gap(y2) + self.gap2(y2)
        y4 = y3.view(y3.shape[0], -1)
        y5 = self.fc1(y4)

        degradation_kernel_list, aux_loss = self.decoder_moe(y5, degradation_kernel, loss_coef=0.02)

        sorted_degradation_kernel_list = sorted(degradation_kernel_list, key=lambda x: (x.size(0), x.size(1)))

        sum_result = None
        for iidx in range(self.start_idx, self.num_experts):
            res_d = self.degradation_filters[iidx - self.start_idx](sr, sorted_degradation_kernel_list[iidx])
            if sum_result is None:
                sum_result = res_d
            else:
                sum_result += res_d

        out = self.conv(sum_result)
        return out, aux_loss


class DeformableAttentionRGB(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DeformableAttentionRGB, self).__init__()

        self.kernel_size = kernel_size
        self.channels_out = channels_out
        self.channels_in = channels_in

        self.deform_conv_rgb = DeformableConvRGB(
            self.channels_in, self.channels_out, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

        self.residual_channel_atn_blk = ResidualChannelAttentionBlock(default_conv, channels_out, 3, reduction)
        self.relu = nn.LeakyReLU(0.1, True)
        self.conv = default_conv(channels_in, channels_out, 3)

    def forward(self, x, inter, fea):
        out1 = self.residual_channel_atn_blk(x)
        out2 = self.deform_conv_rgb(out1, inter, fea)
        out = self.conv(out2 + out1)
        return out


class FusionBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(FusionBlock, self).__init__()
        self.conv1 = default_conv(channels_in, channels_in // 4, 1)
        self.conv2 = default_conv(channels_in, channels_in // 4, 1)
        self.conv3 = default_conv(channels_in // 4, channels_in, 1)
        self.sigmoid = nn.Sigmoid()

        self.conv = default_conv(2 * channels_in, channels_out, 3)

    def forward(self, rgb, dep, inter):
        inter1 = self.conv1(inter)
        rgb1 = self.conv2(rgb)

        w = torch.sigmoid(inter1)
        rgb2 = rgb1 * w
        rgb3 = self.conv3(rgb2) + rgb
        cat1 = torch.cat([rgb3, dep], dim=1)
        out = self.conv(cat1)

        return out


class DOFT(nn.Module):
    """Degradation-Oriented Feature Transform"""

    def __init__(self, channels_in, channels_out, kernel_size, reduction):
        super(DOFT, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size

        self.deform_atn_rgb = DeformableAttentionRGB(channels_in, channels_out, kernel_size, reduction)
        self.fusion_block = FusionBlock(channels_in, channels_out)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x, inter, rgb, fea):
        rgb = self.deform_atn_rgb(rgb, inter, fea)

        out1 = self.fusion_block(rgb, x, inter)
        out = x + out1
        return out


class DepthSuperResolutionNetwork(nn.Module):
    def __init__(self, n_feats=64, reduction=16, conv=default_conv):
        super(DepthSuperResolutionNetwork, self).__init__()

        kernel_size = 3

        n_feats = n_feats

        # head module
        modules_head = [conv(1, n_feats, kernel_size)]
        self.head = nn.Sequential(*modules_head)

        modules_head_rgb = [conv(3, n_feats, kernel_size)]
        self.head_rgb = nn.Sequential(*modules_head_rgb)

        self.dgm1 = DOFT(n_feats, n_feats, 3, reduction)
        self.dgm2 = DOFT(n_feats, n_feats, 3, reduction)
        self.dgm3 = DOFT(n_feats, n_feats, 3, reduction)
        self.dgm4 = DOFT(n_feats, n_feats, 3, reduction)
        self.dgm5 = DOFT(n_feats, n_feats, 3, reduction)

        self.c_d1 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d2 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d3 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)
        self.c_d4 = ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2)

        modules_d5 = [
            conv(5 * n_feats, n_feats, 1),
            ResidualGroup(conv, n_feats, 3, reduction=reduction, n_resblocks=2),
        ]
        self.c_d5 = nn.Sequential(*modules_d5)

        self.c_r1 = conv(n_feats, n_feats, kernel_size)
        self.c_r2 = conv(n_feats, n_feats, kernel_size)
        self.c_r3 = conv(n_feats, n_feats, kernel_size)
        self.c_r4 = conv(n_feats, n_feats, kernel_size)

        self.act = nn.LeakyReLU(0.1, True)

        # tail
        modules_tail = [conv(n_feats, 1, kernel_size)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, inter, rgb, fea):
        # head
        x = self.head(x)
        rgb = self.head_rgb(rgb)
        rgb1 = self.c_r1(rgb)
        rgb2 = self.c_r2(self.act(rgb1))
        rgb3 = self.c_r3(self.act(rgb2))
        rgb4 = self.c_r4(self.act(rgb3))

        dep10 = self.dgm1(x, inter, rgb, fea)
        dep1 = self.c_d1(dep10)
        dep20 = self.dgm2(dep1, inter, rgb1, fea)
        dep2 = self.c_d2(self.act(dep20))
        dep30 = self.dgm3(dep2, inter, rgb2, fea)
        dep3 = self.c_d3(self.act(dep30))
        dep40 = self.dgm4(dep3, inter, rgb3, fea)
        dep4 = self.c_d4(self.act(dep40))
        dep50 = self.dgm5(dep4, inter, rgb4, fea)

        cat1 = torch.cat([dep1, dep2, dep3, dep4, dep50], dim=1)
        dep6 = self.c_d5(cat1)

        res = dep6 + x

        out = self.tail(res)

        return out


class SuperResolutionNetwork(nn.Module):
    def __init__(self, n_feats, reduction):
        super(SuperResolutionNetwork, self).__init__()

        # restorer
        self.restorer = DepthSuperResolutionNetwork(n_feats=n_feats, reduction=reduction)

        # encoder
        self.encoder = DepthEncoder(n_feats=n_feats)

    def forward(self, x_query, rgb):

        fea, degradation_kernel, inter = self.encoder(x_query)
        restored = self.restorer(x_query, inter, rgb, fea)

        return {'restored': restored, 'degradation_kernel': degradation_kernel}


class Net(nn.Module):
    """DDP-enabled base DORNet implementation"""

    def __init__(self, tiny_model=False):
        super(Net, self).__init__()

        if tiny_model:
            n_feats = 24
            reduction = 4
        else:
            n_feats = 64
            reduction = 16

        self.srn = SuperResolutionNetwork(n_feats=n_feats, reduction=reduction)
        self.degradation_router = DegradationRouter(n_feats=n_feats)

        # self.contrastive_loss = ContrastLoss(ablation=False)

    def forward(self, x_query, rgb):

        srn_out = self.srn(x_query, rgb)
        restored = srn_out['restored']
        degradation_kernel = srn_out['degradation_kernel']

        degraded_depth, aux_loss = self.degradation_router(x_query, restored, degradation_kernel)
        # CLLoss1 = self.contrastive_loss(degraded_depth, x_query, restored)

        return {'restored': restored, 'degraded_depth': degraded_depth, 'aux_loss': aux_loss} # , 'CLLoss1': CLLoss1}
